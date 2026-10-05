"""Tests for historical-opponent selection rules."""

import unittest

import numpy as np

from poker_rl.opponent_selection import (
    select_opponent,
    selection_probabilities,
)


class SelectionProbabilityTests(unittest.TestCase):
    def test_latest_selects_newest_opponent(self):
        probabilities = selection_probabilities(
            strategy="latest",
            loss_rates=[0.9, 0.2, 0.5],
            evaluation_counts=[10, 10, 10],
        )

        np.testing.assert_allclose(
            probabilities,
            [0.0, 0.0, 1.0],
        )

    def test_uniform_assigns_equal_probability(self):
        probabilities = selection_probabilities(
            strategy="uniform",
            loss_rates=[0.9, 0.2, 0.5],
            evaluation_counts=[10, 10, 10],
        )

        np.testing.assert_allclose(
            probabilities,
            [1 / 3, 1 / 3, 1 / 3],
        )

    def test_pfsp_prioritizes_larger_weakness(self):
        probabilities = selection_probabilities(
            strategy="pfsp",
            loss_rates=[0.1, 0.8],
            evaluation_counts=[100, 100],
            epsilon=0,
        )

        self.assertGreater(
            probabilities[1],
            probabilities[0],
        )
        self.assertAlmostEqual(
            float(probabilities.sum()),
            1.0,
        )

    def test_uncertainty_prioritizes_less_evaluated_opponent(self):
        probabilities = selection_probabilities(
            strategy="uncertainty",
            loss_rates=[0.5, 0.5],
            evaluation_counts=[100, 1],
            beta=1.0,
            epsilon=0,
        )

        self.assertGreater(
            probabilities[1],
            probabilities[0],
        )
        self.assertAlmostEqual(
            float(probabilities.sum()),
            1.0,
        )

    def test_uniform_mixture_keeps_probability_positive(self):
        probabilities = selection_probabilities(
            strategy="pfsp",
            loss_rates=[0.0, 1.0],
            evaluation_counts=[100, 100],
            epsilon=0.1,
        )

        self.assertGreater(probabilities[0], 0)
        self.assertGreater(probabilities[1], 0)

    def test_seeded_sampling_is_reproducible(self):
        first_generator = np.random.default_rng(42)
        second_generator = np.random.default_rng(42)

        first_samples = [
            select_opponent(
                [0.2, 0.8],
                first_generator,
            )
            for _ in range(20)
        ]
        second_samples = [
            select_opponent(
                [0.2, 0.8],
                second_generator,
            )
            for _ in range(20)
        ]

        self.assertEqual(
            first_samples,
            second_samples,
        )


if __name__ == "__main__":
    unittest.main()