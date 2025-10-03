from collections import deque
from copy import copy
from dataclasses import dataclass

from typing_extensions import override
from z3 import BoolRef, ModelRef, Not, Implies, Or, And, Bool, Solver, sat, ExprRef

from aalpy import Dfa, DfaState
from aalpy.learning_algs.deterministic.Apartness import Apartness
from aalpy.learning_algs.deterministic.ObservationTree import (
    ObservationTree,
    MooreNode,
    MealyNode,
)
from lsharpsat.DCApartness import DCApartness
from lsharpsat.DCValue import common_dc_value
from lsharpsat.ObsTreeVisualizer import visualize_observation_tree
from lsharpsat.logger import get_logger

log = get_logger()


@dataclass
class AutomataFormula:
    formula: list[ExprRef]
    formula_names: list[str]
    col_vars: dict[tuple[tuple, int], BoolRef]
    par_vars: dict[tuple[str, int, int], BoolRef]
    acpt_vars: dict[int, BoolRef]
    alphabet: set[str]

    def create_solver(self) -> Solver:
        solver = Solver()
        solver.set(unsat_core=True)
        for f, n in zip(self.formula, self.formula_names):
            solver.assert_and_track(f, n)
        return solver


def dfa_paths_to_state(dfa: Dfa, state: DfaState):
    """Generator of all paths from from_state to state in dfa, BFS"""
    queue = [([], dfa.initial_state)]

    while queue:
        path, current_state = queue.pop(0)
        if current_state == state:
            yield path

        for input_val, next_state in current_state.transitions.items():
            queue.append((path + [input_val], next_state))


def _words_from_node(node: MooreNode, alphabet: set[str]) -> set[tuple[str]]:
    words = {tuple()}
    for a in alphabet:
        if succ := node.get_successor(a):
            words.update({(a,) + w for w in _words_from_node(succ, alphabet)})

    return words


def word_of_node(node: MooreNode) -> tuple:
    if node.parent is None:
        return tuple()
    return word_of_node(node.parent) + (node.input_to_parent,)


def not_model(model: ModelRef, formula: AutomataFormula) -> ExprRef:
    clauses = []
    for v in formula.col_vars.values():
        clauses.append(v != model.eval(v))

    for v in formula.par_vars.values():
        clauses.append(v != model.eval(v))

    for v in formula.acpt_vars.values():
        clauses.append(v != model.eval(v))

    return Or(clauses)


def build_dfa_from_model(formula: AutomataFormula, model: ModelRef) -> Dfa[str]:
    state_count = len(formula.acpt_vars)
    initial_state = None
    states = {}

    for s in range(state_count):
        is_accepting = bool(model.eval(formula.acpt_vars[s]))
        state_id = f"{s}"
        # words = []
        # for (w, q), cv in formula.col_vars.items():
        #     if q != s:
        #         continue
        #     if model.eval(cv):
        #         words.append(str(w))
        # state_id += ": " + ', '.join(words)
        state = DfaState(state_id=state_id, is_accepting=is_accepting)
        states[s] = state

        if model.eval(formula.col_vars[((), s)]):
            if initial_state is not None:
                raise ValueError("Multiple initial states in model")
            initial_state = state

    for s in range(state_count):
        state = states[s]
        for dest in range(state_count):
            for a in formula.alphabet:
                if model.eval(formula.par_vars[(a, s, dest)]):
                    state.transitions[a] = states[dest]

    if initial_state is None:
        raise ValueError("No initial state in model")

    return Dfa(initial_state, list(states.values()))


class DCObservationTree(ObservationTree):
    apartness = DCApartness

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_hyp_size = 0

    def _words(self) -> list[tuple[str]]:
        return sorted(_words_from_node(self.root, self.alphabet))

    def _build_z3_formula(self, extra_states: int) -> AutomataFormula:
        n_colors = len(self.basis) + extra_states
        words = self._words()
        basis = [word_of_node(b) for b in self.basis]
        # log.info(f"{words}")

        col_vars = {}
        for w in words:
            for c in range(n_colors):
                col_vars[(w, c)] = Bool(f"c_{{{w},{c}}}")

        par_vars = {}
        for l in self.alphabet:
            for c_src in range(n_colors):
                for c_dst in range(n_colors):
                    par_vars[(l, c_src, c_dst)] = Bool(f"p_{{{l},{c_src},{c_dst}}}")

        acpt_vars = {}
        for c in range(n_colors):
            acpt_vars[c] = Bool(f"a_{c}")

        formula = []
        formula_names = []

        # 0a: Set basis states to set colors
        for i, b in enumerate(basis):
            formula.append(col_vars[(b, i)])
            formula_names.append(f"basis_{b}_color_{i}")

        # 0b: Set identified frontier states to basis color
        # for f, bs in self.frontier_to_basis_dict.items():
        #     if len(bs) == 1:
        #         b = bs[0]
        #         i = basis.index(word_of_node(b))
        #         w = word_of_node(f)
        #         formula.append(col_vars[(w, i)])
        #         formula_names.append(f"frontier_{w}_to_basis_{b}_({i})")

        # 1,2: Set color to accepting or rejecting
        for w in words:
            for c in range(n_colors):
                if (
                    len(w) == 0
                    and self.root.output
                    or len(w) > 0
                    and self.get_observation(w)[-1]
                ):
                    a_r = acpt_vars[c]
                else:
                    a_r = Not(acpt_vars[c])
                formula.append(Implies(col_vars[(w, c)], a_r))
                formula_names.append(f"state_{w}_color_{c}_is_accepting_{a_r}")

        # 3: Each word has at least one color
        for w in words:
            disj = []
            for c in range(n_colors):
                disj.append(col_vars[(w, c)])
            formula.append(Or(disj))
            formula_names.append(f"state_{w}_has_color")

        # 4: when a state and its parent are colored the transition is set
        for c_src in range(n_colors):
            for c_dst in range(n_colors):
                for w in words:
                    if len(w) > 0:
                        formula.append(
                            Implies(
                                And(col_vars[(w[:-1], c_src)], col_vars[(w, c_dst)]),
                                par_vars[w[-1], c_src, c_dst],
                            )
                        )
                        formula_names.append(
                            f"state_{w}_color_{c_dst}_and_parent_color_{c_src}_implies_transition_{w[-1]}_{c_src}_{c_dst}"
                        )

        # 5: Each transition can only target one state
        for l in self.alphabet:
            for c_src in range(n_colors):
                for c_dst in range(n_colors):
                    for c_dst_other in range(c_dst + 1, n_colors):
                        formula.append(
                            Or(
                                Not(par_vars[(l, c_src, c_dst)]),
                                Not(par_vars[(l, c_src, c_dst_other)]),
                            )
                        )
                        formula_names.append(
                            f"transition_{l}_{c_src}_to_{c_dst}_and_{c_dst_other}_mutually_exclusive"
                        )

        # 6: Each state has at most one color
        for w in words:
            for c in range(n_colors):
                for c_prime in range(c + 1, n_colors):
                    formula.append(
                        Or(Not(col_vars[(w, c)]), Not(col_vars[(w, c_prime)]))
                    )
                    formula_names.append(
                        f"state_{w}_color_{c}_and_{c_prime}_mutually_exclusive"
                    )

        # 7: Each transition must target at least one state
        for l in self.alphabet:
            for c_src in range(n_colors):
                disj = []
                for c_dst in range(n_colors):
                    disj.append(par_vars[(l, c_src, c_dst)])
                formula.append(Or(disj))
                formula_names.append(f"transition_{l}_{c_src}_has_target")

        # 8: State color is set when transition and parent color are set
        for w in words:
            if len(w) == 0:
                continue

            for c_src in range(n_colors):
                for c_dst in range(n_colors):
                    formula.append(
                        Implies(
                            And(
                                par_vars[(w[-1], c_src, c_dst)],
                                col_vars[(w[:-1], c_src)],
                            ),
                            col_vars[(w, c_dst)],
                        )
                    )
                    formula_names.append(
                        f"state_{w}_color_{c_dst}_is_set_by_parent_color_{c_src}_and_transition_{w[-1]}_{c_src}_{c_dst}"
                    )

        # 9: Two apart states must be different
        for c in range(n_colors):
            for w in words:
                s_w = self.get_successor(w)
                for v in words:
                    if w == v:
                        continue

                    s_v = self.get_successor(v)
                    if Apartness.states_are_apart(s_w, s_v, self):
                        formula.append(Implies(col_vars[(w, c)], Not(col_vars[(v, c)])))
                        formula_names.append(
                            f"states_{w}_and_{v}_are_apart_cannot_share_color_{c}"
                        )

        return AutomataFormula(
            formula, formula_names, col_vars, par_vars, acpt_vars, self.alphabet
        )

    def _find_basis_states_in_hypothesis(self, hypothesis: Dfa):
        self.states_dict = dict()
        for b in self.basis:
            w = word_of_node(b)
            hypothesis.reset_to_initial()
            for l in w:
                hypothesis.step(l)
            self.states_dict[b] = hypothesis.current_state

        for s in hypothesis.states:
            if s not in self.states_dict.values():
                for path in dfa_paths_to_state(hypothesis, s):
                    tree_node = self.get_successor(path)
                    if tree_node is not None:
                        self.states_dict[tree_node] = s
                        break

    @override
    def make_observation_tree_adequate(self):
        self.update_frontier_and_basis()
        old_frontier_to_basic_dict = None
        while (
            not self.is_observation_tree_adequate()
            and old_frontier_to_basic_dict != self.frontier_to_basis_dict
        ):
            old_frontier_to_basic_dict = copy(self.frontier_to_basis_dict)
            self.make_basis_complete()
            self.make_frontiers_identified()
            self.promote_frontier_state()

    def get_distinguishing_sequences(self, group: list[MooreNode | MealyNode]):
        """
        Get distinguishing sequences for a group of states.
        :param group: list of states
        :return: generator of distinguishing sequences
        """
        if self.automaton_type == "mealy":
            return self._get_distinguishing_sequences_mealy(group)
        else:
            return self._get_distinguishing_sequences_moore(group)

    def _get_distinguishing_sequences_mealy(self, group: list[MealyNode]):
        # Identifies all distinguishing input-output pairs in the provided alphabet of the n states
        groups = deque([([], group)])

        while groups:
            access_seq, group = groups.popleft()
            for input_val in self.alphabet:
                valid_group = [node for node in group if input_val in node.successors]

                if len(valid_group) >= 2:
                    outputs = [node.get_output(input_val) for node in valid_group]
                    try:
                        common_dc_value(*outputs)
                    except ValueError:
                        yield access_seq + [input_val]

                    groups.append(
                        (
                            access_seq + [input_val],
                            [node.get_successor(input_val) for node in valid_group],
                        )
                    )

    def _get_distinguishing_sequences_moore(self, group: list[MooreNode]):
        # Identifies if two states can be distinguished by any input-output pair in the provided alphabet
        groups = deque([([], group)])

        while groups:
            access_seq, group = groups.popleft()
            outputs = [node.output for node in group]
            try:
                common_dc_value(*outputs)
            except ValueError:
                yield access_seq

            for input_val in self.alphabet:
                successors = [
                    s
                    for s in [node.get_successor(input_val) for node in group]
                    if s is not None
                ]
                if len(successors) >= 2:
                    groups.append((access_seq + [input_val], successors))

    @override
    def identify_frontier(self, frontier_state):
        if frontier_state not in self.frontier_to_basis_dict:
            raise Exception(
                f"Warning: {frontier_state} not found in frontier_to_basis_dict."
            )

        self.update_basis_candidates(frontier_state)
        old_candidate_size = len(self.frontier_to_basis_dict.get(frontier_state))
        if old_candidate_size < 2:
            return

        if self.separation_rule == "SepSeq" or old_candidate_size == 2:
            self._identify_frontier_sepseq(frontier_state)
        else:
            inputs, outputs = self._identify_frontier_ads(frontier_state)
            self.insert_observation(inputs, outputs)

        self.update_basis_candidates(frontier_state)
        if (
            len(self.frontier_to_basis_dict.get(frontier_state)) == old_candidate_size
            and len(self.frontier_to_basis_dict[frontier_state]) > 1
        ):
            # Because of DCs we should not raise an exception here anymore
            log.debug(f"Identification did not help for {frontier_state.id}")
            new_info = False
            for inp in self.alphabet:
                if frontier_state.get_successor(inp) is None:
                    new_info = True
                    inputs = self.get_transfer_sequence(self.root, frontier_state) + [
                        inp
                    ]
                    outputs = self.sul.query(inputs)
                    self.insert_observation(inputs, outputs)

            if not new_info:
                log.warning(
                    f"Identification did not help for {frontier_state.id}, and no new information could be added."
                )
            else:
                log.info(f"Retrying identification for {frontier_state.id}")
                self.identify_frontier(frontier_state)

    @override
    def _identify_frontier_sepseq(self, frontier_state):
        # Specifically identify frontier states using separating sequences
        basis_candidates = self.frontier_to_basis_dict.get(frontier_state)

        witnesses = [
            w
            for w in self.get_distinguishing_sequences(basis_candidates)
            if self.get_successor(w, from_node=frontier_state) is None
        ]

        if len(witnesses) == 0:
            log.debug(
                f"No possible witnesses found for {frontier_state.id} with basis {[b.id for b in basis_candidates]}"
            )

        for witness in witnesses:
            inputs = self.get_transfer_sequence(self.root, frontier_state)
            inputs.extend(witness)
            outputs = self.sul.query(inputs)

            self.insert_observation(inputs, outputs)

    @override
    def construct_hypothesis(self):
        visualize_observation_tree(
            self, f"out/ob_tree_{len(self.basis)}_basis.png", file_type="png"
        )

        extra_states = max(0, self.last_hyp_size - len(self.basis))
        old_extra_states = None

        hypotheses = []

        while True:
            log.debug(
                f"Trying to build dfa with {len(self.basis) + extra_states} states"
            )
            if len(self.basis) + extra_states > 6:
                log.error("Something went wrong, could not find the dfa")
                exit(1)

            if old_extra_states != extra_states:
                formula = self._build_z3_formula(extra_states)
                old_extra_states = extra_states
                solver = formula.create_solver()

            log.info("Solving...")
            # log.debug(f"Solving {solver.sexpr()}")

            if solver.check() == sat:
                m = solver.model()
                hypothesis = build_dfa_from_model(formula, m)
                self._find_basis_states_in_hypothesis(hypothesis)
                hypothesis.visualize(
                    f"out/hypothesis_dfa_{hypothesis.size}_{len(hypotheses)}.png",
                    file_type="png",
                )
                hypothesis.compute_prefixes()
                hypothesis.characterization_set = (
                    hypothesis.compute_characterization_set(raise_warning=False)
                )
                log.info(f"sat, found dfa {len(hypotheses)}")
                solver.add(not_model(m, formula))
                hypotheses.append(hypothesis)
                self.last_hyp_size = len(hypothesis.states)
                return hypothesis
            else:
                log.info("unsat")
                c = solver.unsat_core()
                log.info(f"unsat core (len {len(c)}): {c}")
                # log.debug(f"with formula {solver.sexpr()}")
                if len(hypotheses) > 0:
                    log.info(f"found {len(hypotheses)} hypotheses")
                    return hypotheses[-1]
                else:
                    extra_states += 1

    @override
    def get_successor(self, inputs, from_node=None):
        """
        Retrieve the node (subtree) corresponding to the given input sequence, given an optional starting node.
        """
        if from_node is None:
            current_node = self.root
        else:
            current_node = from_node
        for input_val in inputs:
            successor_node = current_node.get_successor(input_val)
            if successor_node is None:
                return None
            current_node = successor_node
        return current_node

    def _get_output_sequence(self, inputs, query_mode="full"):
        """
        Returns the sequence of outputs corresponding to the input path.
        The knowledge is obtained from the observation tree or if not available via querying the sul.
        There are 3 query_modes: full, none and final. They allow you to restrict the querying to your needs
        """
        assert query_mode in ["full", "none", "final"]
        if self.automaton_type != "dfa":
            return self.sul.query(inputs)
        else:
            outputs = []
            current_node = self.root
            for inp_num in range(len(inputs)):
                inp = inputs[inp_num]
                if current_node is not None:
                    current_node = current_node.get_successor(inp)
                if current_node is None:
                    if query_mode == "full" or (
                        inp_num == len(inputs) - 1 and query_mode == "final"
                    ):
                        outputs.append(self.sul.query(inputs[: inp_num + 1]))
                    else:
                        outputs.append(None)
                else:
                    if current_node.output is None and (
                        query_mode == "full"
                        or (inp_num == len(inputs) - 1 and query_mode == "final")
                    ):
                        outputs.append(self.sul.query(inputs[: inp_num + 1]))
                    else:
                        outputs.append(current_node.output)
            return outputs

    # @override
    # def _process_binary_search(self, hypothesis, cex_inputs, cex_outputs):
    #     """
    #     use LINEAR search on the counter example to compute a witness between the real system and the hypothesis
    #     override the binary search from the parent class
    #     """
    #     visualize_observation_tree(
    #         self, f"out/ob_tree_{len(self.basis)}_ce.png", file_type="png"
    #     )
    #
    #     nodes_dict = {}
    #     for hyp_node, hyp_state in self.states_dict.items():
    #         nodes_dict[hyp_state] = hyp_node
    #
    #     access_seq = cex_inputs
    #     tree_node = self.get_successor(cex_inputs)
    #     witness_seq = []
    #     while not (tree_node in self.frontier_to_basis_dict or tree_node in self.basis):
    #         witness_seq = [access_seq[-1]] + witness_seq
    #         access_seq = access_seq[:-1]
    #
    #         tree_node = tree_node.parent
    #         hyp_state = self._get_automaton_successor(
    #             hypothesis, hypothesis.initial_state, access_seq
    #         )
    #         hyp_node = nodes_dict[hyp_state]
    #         hyp_access = self.get_transfer_sequence(self.root, hyp_node)
    #
    #         hyp_output = self._get_automaton_successor(
    #             hypothesis, hyp_state, witness_seq
    #         ).output
    #
    #         witness_node = self.get_successor(witness_seq, from_node=hyp_node)
    #         if witness_node is None or witness_node.output is None:
    #             output_seq = self.sul.query(hyp_access + witness_seq)
    #             self.insert_observation(hyp_access + witness_seq, output_seq)
    #             witness_node = self.get_successor(witness_seq, from_node=hyp_node)
    #             visualize_observation_tree(
    #                 self, f"out/ob_tree_{len(self.basis)}_ce.png", file_type="png"
    #             )
    #         else:
    #             # No new information will be inserted, since node is already explored.
    #             # Either the node is consistent with hypothesis,
    #             # or it has been inserted during the counter example processing.
    #             # To prevent looping we ignore it.
    #             continue
    #
    #         tree_output = witness_node.output
    #
    #         if hyp_output != tree_output:
    #             access_seq = hyp_access + witness_seq
    #             tree_node = self.get_successor(access_seq)
    #             witness_seq = []

    @override
    def _process_binary_search(self, hypothesis, cex_inputs, cex_outputs):
        """
        use binary search on the counter example to compute a witness between the real system and the hypothesis
        """
        visualize_observation_tree(
            self, f"out/ob_tree_{len(self.basis)}_ce.png", file_type="png"
        )
        tree_node = self.get_successor(cex_inputs)
        self.update_frontier_and_basis()

        if tree_node in self.frontier_to_basis_dict or tree_node in self.basis:
            return

        hyp_state = self._get_automaton_successor(
            hypothesis, hypothesis.initial_state, cex_inputs
        )
        hyp_node = list(self.states_dict.keys())[
            list(self.states_dict.values()).index(hyp_state)
        ]

        prefix = []
        current_state = self.root
        for input in cex_inputs:
            if current_state in self.frontier_to_basis_dict:
                break
            current_state = current_state.get_successor(input)
            prefix.append(input)

        h = (len(prefix) + len(cex_inputs)) // 2
        sigma1 = list(cex_inputs[:h])
        sigma2 = list(cex_inputs[h:])

        hyp_state_p = self._get_automaton_successor(
            hypothesis, hypothesis.initial_state, sigma1
        )
        hyp_node_p = list(self.states_dict.keys())[
            list(self.states_dict.values()).index(hyp_state_p)
        ]
        hyp_p_access = self.get_transfer_sequence(self.root, hyp_node_p)

        if not self.apartness.states_are_apart(tree_node, hyp_node, self):
            witness = []
        else:
            witness = self.apartness.compute_witness(tree_node, hyp_node, self)
            if witness is None:
                raise RuntimeError("Binary search: There should be a witness")

        query_inputs = hyp_p_access + sigma2 + witness
        query_outputs = self.sul.query(query_inputs)

        self.insert_observation(query_inputs, query_outputs)
        visualize_observation_tree(
            self, f"out/ob_tree_{len(self.basis)}_ce.png", file_type="png"
        )

        tree_node_p = self.get_successor(sigma1)

        witness_p = self.apartness.compute_witness(tree_node_p, hyp_node_p, self)

        if witness_p is not None:
            self._process_binary_search(hypothesis, sigma1, cex_outputs[:h])
        else:
            new_inputs = list(hyp_p_access) + sigma2
            print(
                f"New inputs: {new_inputs}, outputs: {query_outputs[: len(new_inputs)]}"
            )
            self._process_binary_search(
                hypothesis, new_inputs, query_outputs[: len(new_inputs)]
            )
