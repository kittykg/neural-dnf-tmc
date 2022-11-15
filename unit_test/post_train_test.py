import sys
import unittest

import torch

sys.path.append("../")

from dnf_post_train import (
    remove_unused_conjunctions,
    remove_disjunctions_when_empty_conjunctions,
)
from rule_learner import DNFClassifier


class TestPostTrainMethods(unittest.TestCase):
    def test_remove_unused_conjunctions_1(self):
        test_model = DNFClassifier(4, 3, 2)
        test_model.dnf.disjunctions.weights.data = torch.Tensor(
            [[1, 0, -1], [0, 1, 0]]
        )  # 2 x 3 matrix
        og_conj_weight = torch.randint(-1, 2, (3, 4))  # 3 x 4 matrix
        test_model.dnf.conjunctions.weights.data = og_conj_weight

        unused_count = remove_unused_conjunctions(test_model)

        self.assertEqual(unused_count, 0)
        torch.testing.assert_close(  # type: ignore
            test_model.dnf.conjunctions.weights.data, og_conj_weight
        )

    def test_remove_unused_conjunctions_2(self):
        test_model = DNFClassifier(4, 3, 2)
        test_model.dnf.disjunctions.weights.data = torch.Tensor(
            [[1, 0, 0], [0, 0, 0]]
        )  # 2 x 3 matrix
        og_conj_weight = torch.randint(-1, 2, (3, 4))  # 3 x 4 matrix
        test_model.dnf.conjunctions.weights.data = og_conj_weight

        unused_count = remove_unused_conjunctions(test_model)

        expected_new_conj_weight = og_conj_weight
        expected_new_conj_weight[-1, :] = 0
        expected_new_conj_weight[1, :] = 0

        self.assertEqual(unused_count, 2)
        torch.testing.assert_close(  # type: ignore
            test_model.dnf.conjunctions.weights.data, og_conj_weight
        )

    def test_remove_disjunctions_when_empty_conjunctions_1(self):
        test_model = DNFClassifier(4, 3, 2)
        test_model.dnf.conjunctions.weights.data = torch.Tensor(
            [[1, 0, -1, 1], [0, 0, 1, 0], [0, 0, 1, 1]]
        )  # 3 x 4 matrix
        og_disj_weight = torch.randint(-1, 2, (2, 3))  # 2 x 3 matrix
        test_model.dnf.disjunctions.weights.data = og_disj_weight

        unused_count = remove_disjunctions_when_empty_conjunctions(test_model)

        self.assertEqual(unused_count, 0)
        torch.testing.assert_close(  # type: ignore
            test_model.dnf.disjunctions.weights.data, og_disj_weight
        )

    def test_remove_disjunctions_when_empty_conjunctions_2(self):
        test_model = DNFClassifier(4, 3, 2)
        test_model.dnf.conjunctions.weights.data = torch.Tensor(
            [[1, 0, -1, 1], [0, 0, 0, 0], [0, 0, 1, 1]]
        )  # 3 x 4 matrix
        og_disj_weight = torch.randint(-1, 2, (2, 3))  # 2 x 3 matrix
        test_model.dnf.disjunctions.weights.data = og_disj_weight

        unused_count = remove_disjunctions_when_empty_conjunctions(test_model)

        expected_new_disj_weight = og_disj_weight
        expected_new_disj_weight.T[1, :] = 0

        self.assertEqual(unused_count, 2)
        torch.testing.assert_close(  # type: ignore
            test_model.dnf.disjunctions.weights.data, expected_new_disj_weight
        )


if __name__ == "__main__":
    unittest.main()
