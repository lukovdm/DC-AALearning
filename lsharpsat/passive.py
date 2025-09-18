import random

from aalpy import compare_automata
from aalpy.learning_algs.deterministic.ObservationTree import ObservationTree
from aalpy.utils import generate_random_deterministic_automata
from aalpy.automata.Dfa import Dfa
from aalpy.SULs.AutomataSUL import AutomatonSUL
from aalpy.oracles.RandomWalkEqOracle import RandomWalkEqOracle
from lsharpsat.DCLSharp import run_Lsharp
from lsharpsat.DCSUL import RandomDCSUL, OutputDCSUL

from lsharpsat.logger import get_logger, setup_logger


DFA_SIZE = 5

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
        output_alphabet_size=3
    )

    random_dfa.visualize("out/random_dfa.png", file_type="png")

    alphabet = random_dfa.get_input_alphabet()

    sul = OutputDCSUL(AutomatonSUL(random_dfa), "o3", {"o1": True, "o2": False})
    eq_oracle = RandomWalkEqOracle(alphabet, sul, num_steps=5000, reset_prob=0.09)

    found_dfa = run_Lsharp(alphabet, sul, eq_oracle, "dfa", print_level=3)

    found_dfa.visualize("out/found_dfa.png", file_type="png")
    correct = compare_automata(random_dfa, found_dfa)
    if len(correct) == 0:
        log.info("Successfully learned the correct automaton")
    else:
        log.error("The learned automaton is NOT equivalent to the original one")
        for w in correct:
            log.error(f"Counterexample: {w}")


if __name__ == "__main__":
    setup_logger(level="DEBUG")
    test_random_dfa()
