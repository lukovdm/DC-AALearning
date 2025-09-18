from __future__ import annotations

from aalpy.learning_algs.deterministic.ObservationTree import (
    ObservationTree,
    MealyNode,
    MooreNode,
)
from aalpy.automata import (
    Dfa,
    DfaState,
    MealyMachine,
    MealyState,
    MooreMachine,
    MooreState,
)

_BASIS_COLORS = [str(c) for c in range(1, 13)]


def _collect_tree_nodes(root) -> list:  # BFS collection of all nodes
    visited = []
    queue = [root]
    seen = set()
    while queue:
        n = queue.pop(0)
        if id(n) in seen:
            continue
        seen.add(id(n))
        visited.append(n)
        if isinstance(n, MealyNode):
            for _, (_, succ) in n.successors.items():
                queue.append(succ)
        else:
            assert isinstance(n, MooreNode)
            for _, succ in n.successors.items():
                queue.append(succ)
    return visited


def _assign_basis_colors(basis_nodes: list) -> dict:
    mapping = {}
    for i, b in enumerate(basis_nodes):
        mapping[b] = _BASIS_COLORS[i % len(_BASIS_COLORS)]
    return mapping


def _build_raw_tree_automaton(obs_tree: ObservationTree):
    aut_type = obs_tree.automaton_type
    root = obs_tree.root
    nodes = _collect_tree_nodes(root)

    basis_nodes = list(obs_tree.basis)
    color_map = _assign_basis_colors(basis_nodes)

    state_map = {}
    for node in nodes:
        num = node.id
        if node in basis_nodes:
            color = color_map[node]
            shape = "octagon"
        elif node in obs_tree.frontier_to_basis_dict:
            cands = obs_tree.frontier_to_basis_dict[node]
            if len(cands) == 1:
                # single candidate frontier
                color = color_map[next(iter(cands))]
                shape = "circle"
            else:
                # multiple candidates frontier, pick first candidate color if exists
                color = "9"
                shape = "circle"
        else:
            color = "white"
            shape = "circle"

        state_id = f"{num}~{color}%{shape}"

        if aut_type == "mealy":
            st = MealyState(state_id)
        elif aut_type == "moore":
            st = MooreState(state_id, output=node.output)
        else:
            assert aut_type == "dfa"
            st = DfaState(state_id)
            st.is_accepting = getattr(node, "output", False)
            if getattr(st.is_accepting, "is_dc", False):
                st.state_id = f"{num}~{color}:4%{shape}"
        state_map[node] = st

    # transitions
    for node, st in state_map.items():
        st.transitions = {}
        if aut_type == "mealy":
            st.output_fun = {}
            for inp, (out, succ) in node.successors.items():
                st.transitions[inp] = state_map[succ]
                st.output_fun[inp] = out
        else:
            for inp, succ in node.successors.items():
                st.transitions[inp] = state_map[succ]

    initial_state = state_map[root]

    if aut_type == "mealy":
        automaton = MealyMachine(initial_state, list(state_map.values()))
    elif aut_type == "moore":
        automaton = MooreMachine(initial_state, list(state_map.values()))
    else:
        automaton = Dfa(initial_state, list(state_map.values()))
    return automaton


def visualize_observation_tree(
    obs_tree: ObservationTree,
    path: str = "ObservationTree",
    file_type: str = "pdf",
    display_same_state_transitions: bool = True,
):
    automaton = _build_raw_tree_automaton(obs_tree)
    automaton.visualize(
        path=path,
        file_type=file_type,
        display_same_state_transitions=display_same_state_transitions,
    )
    return automaton


__all__ = ["visualize_observation_tree"]
