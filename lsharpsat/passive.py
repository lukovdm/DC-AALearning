import random

from aalpy import compare_automata
from aalpy.learning_algs.deterministic.ObservationTree import ObservationTree
from aalpy.utils import generate_random_deterministic_automata
from aalpy.automata.Dfa import Dfa
from aalpy.SULs.AutomataSUL import AutomatonSUL
from aalpy.oracles.RandomWalkEqOracle import RandomWalkEqOracle
from lsharpsat.DCLSharp import run_Lsharp
from lsharpsat.DCSUL import RandomDCSUL, OutputDCSUL
from lsharpsat.DCValue import DCValue

from lsharpsat.logger import get_logger, setup_logger


DFA_SIZE = 6

log = get_logger()


def do_n_obs_tree_steps(obs_tree: ObservationTree, num_steps: int):
    obs_tree.update_frontier_and_basis()
    for _ in range(num_steps):
        obs_tree.make_basis_complete()
        obs_tree.make_frontiers_identified()
        obs_tree.promote_frontier_state()


def test_random_dfa():
    log.info("Starting random dfa test")

    random.seed(0)
    random_dfa: Dfa = generate_random_deterministic_automata(
        automaton_type="moore",
        num_states=DFA_SIZE,
        input_alphabet_size=3,
        output_alphabet_size=3,
        # custom_output_alphabet=[DCValue(True), DCValue(False), DCValue()],
    )

    random_dfa.visualize("out/random_dfa.png", file_type="png")

    alphabet = random_dfa.get_input_alphabet()

    sul = OutputDCSUL(AutomatonSUL(random_dfa), "o3", {"o1": True, "o2": False})
    # sul = AutomatonSUL(random_dfa)
    eq_oracle = RandomWalkEqOracle(alphabet, sul, num_steps=5000, reset_prob=0.09)

    found_dfa = run_Lsharp(
        alphabet, sul, eq_oracle, "dfa", print_level=3, extension_rule=None
    )

    found_dfa.visualize("out/found_dfa.png", file_type="png")

    output_mapping = {"o1": True, "o2": False, "o3": DCValue.DC}
    for _ in range(10):
        random_dfa.reset_to_initial()
        found_dfa.reset_to_initial()
        inputs = [random.choice(alphabet) for _ in range(10)]
        random_path = [random_dfa.initial_state]
        found_path = [found_dfa.initial_state]
        for i in inputs:
            random_dfa.step(i)
            found_dfa.step(i)
            random_path.append(random_dfa.current_state)
            found_path.append(found_dfa.current_state)

        # Print paths underneath each other aligned
        print("Inputs: ", " ".join(inputs))
        print(
            "Random DFA states: ",
            " -> ".join(
                [f"{s.state_id} {output_mapping[s.output]}" for s in random_path]
            ),
        )
        print(
            "Found DFA states:  ",
            " -> ".join([f"{s.state_id} {s.output}" for s in found_path]),
        )
        print()


if __name__ == "__main__":
    setup_logger(level="DEBUG")
    test_random_dfa()
