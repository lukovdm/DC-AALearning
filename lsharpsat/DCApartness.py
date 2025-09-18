import logging
from typing import Optional, override

from aalpy.learning_algs.deterministic.Apartness import Apartness
from aalpy.learning_algs.deterministic.ObservationTree import MooreNode, MealyNode
from lsharpsat import logger
from lsharpsat.DCValue import DCValue

log = logger.get_logger()


class DCApartness(Apartness):

    @staticmethod
    def clone_subtree[NT: MealyNode | MooreNode](node: NT, access: list) -> NT:
        if isinstance(node, MooreNode):
            new_node = MooreNode(node.parent)
            new_node.successors = {}
            for k, v in node.successors.items():
                new_node.successors[k] = DCApartness.clone_subtree(v, access + [k])
                new_node.successors[k].parent = new_node
                new_node.successors[k].input_to_parent = k
            new_node.output = node.output
        else:
            assert isinstance(node, MealyNode)
            new_node = MealyNode(node.parent)
            new_node.successors = {}
            for k, (out, v) in node.successors.items():
                new_node.successors[k] = (
                    out,
                    DCApartness.clone_subtree(v, access + [k]),
                )
                new_node.successors[k][1].parent = new_node
                new_node.successors[k][1].input_to_parent = k

        new_node.access_sequence = access
        return new_node

    @staticmethod
    def get_successors[NT](node: NT, input_list: list) -> Optional[NT]:
        for i in input_list:
            if node is None:
                return None
            node = node.get_successor(i)
        return node

    @staticmethod
    @override
    def states_are_apart(first, second, ob_tree):
        # Checking apartness is easier than checking incompatibility
        if Apartness.states_are_apart(first, second, ob_tree):
            return True

        # Assumes that a node cannot be a descendant of a node with a higher id
        if second.id < first.id:
            first, second = second, first

        # Unfortunately, we need to clone the tree to avoid modifying it.
        # This is very slow (dominates the running time), so I am looking for a better solution.
        # It works for testing the amount of queries though.
        # The actual merging takes about the same time as the apartness checking over the course of the whole algorithm.
        first_input = ob_tree.get_access_sequence(first)
        second_input = ob_tree.get_access_sequence(second)
        root = DCApartness.clone_subtree(ob_tree.root, [])
        first_node = DCApartness.get_successors(root, first_input)
        second_node = DCApartness.get_successors(root, second_input)

        if first_node is None or second_node is None:
            raise ValueError("Could not find cloned nodes in the cloned tree.")

        # Try merging the two nodes, and see if there is a conflict.
        # In case of a conflict, we get the access sequences to the nodes causing the conflict
        res = DCApartness.merge(first_node, second_node)

        if res is not None:
            first_access, second_access = res
            log.debug(
                f"Merging nodes {first.id} and {second.id} caused a conflict: {res}."
            )

            # Construct possible candidates that can prove apartness.
            # The first candidate is the transfer sequence from the first node to the first node causing the conflict
            candidate = first_access[len(first_node.access_sequence) :]
            candidates = [candidate]

            # The other candidates are given by extending the candidate while walking backwards over the tree
            # from the second node causing the conflict, until we reach the second node.
            # This assumes that the first candidate is a suffix of the second access sequence,
            # but that seems to hold.
            while candidate != second_access[len(second_node.access_sequence) :]:
                candidate = [second_access[-len(candidate) - 1]] + candidate
                candidates.append(candidate)

            # From the list of candidates, we can construct the experiments.
            # The pairs are given by simply appending the candidates to the two nodes.
            # For now, we already do the experiments here.
            # In theory, you can stop once an experiment shows apartness.
            for candidate in candidates:
                res1 = ob_tree.experiment(first_node.access_sequence + candidate)
                res2 = ob_tree.experiment(second_node.access_sequence + candidate)
                log.debug("Experiment", res1, res2)

            if Apartness.states_are_apart(first, second, ob_tree):
                log.info("States are apart after experiments")
            else:
                log.warning("States are NOT apart after experiments")
            return True

        # Compatible!
        return False

    @staticmethod
    def merge[NT: MooreNode | MealyNode](first: NT, second: NT):
        """
        Merge the second node into the first node.
        :param first: Node to merge into
        :param second: Node to merge from
        :return: Whether there was a conflict during the merge
        """

        # Prevent merging a node with itself
        if first.id == second.id:
            return None

        # Update the output of the first node,
        # while ensuring local compatibility
        if first.output != second.output:
            return first.access_sequence, second.access_sequence
        elif isinstance(first.output, DCValue) and first.output.is_dc:
            first.output = second.output

        # When merging two nodes, we might create a non-deterministic automaton.
        # To solve this, we first recursively merge the nodes that would create non-determinism.
        while True:
            for input_val in second.successors.keys():
                if (
                    input_val in first.successors
                    and first.successors[input_val].id
                    != second.successors[input_val].id
                ):
                    # Nodes share a common successor, so we need to merge those first
                    res = DCApartness.merge(
                        first.successors[input_val], second.successors[input_val]
                    )
                    if res is not None:
                        # Merging successors led to a conflict
                        return res
                    break
            else:
                # No more common successors
                break

        # From this point on, we can assume that merges will not lead to a non-deterministic automaton,
        # so we can simply copy the successors from the second node to the first node.
        # Note that we don't actually use the "parent" attribute anywhere, so we don't need to update that.
        for input_val in second.successors.keys():
            first.successors[input_val] = second.successors[input_val]

        first.id = f"{first.id}+{second.id}"

        # Make second object point to first instead
        second.id = first.id
        second.output = first.output
        second.successors = first.successors
        return None
