"""Tests for opponent-pool training configuration."""

import unittest

from poker_rl.pool_config import (
    PoolTrainConfig,
    pool_config_as_dict,
)


class PoolTrainConfigTests(unittest.TestCase):
    def test_valid_configuration(self):
        config = PoolTrainConfig(
            strategy="uncertainty",
            seed=7,
            budget_seconds=300,
            checkpoint_seconds=30,
            evaluation_games=100,
        )

        self.assertEqual(
            config.strategy,
            "uncertainty",
        )
        self.assertEqual(config.seed, 7)
        self.assertEqual(config.max_pool_size, 10)

    def test_serialization_identifies_experiment(self):
        config = PoolTrainConfig(
            strategy="pfsp",
            seed=1,
            budget_seconds=60,
        )

        serialized = pool_config_as_dict(config)

        self.assertEqual(
            serialized["algorithm"],
            "dqn",
        )
        self.assertEqual(
            serialized["experiment"],
            "historical_opponent_pool",
        )
        self.assertEqual(
            serialized["strategy"],
            "pfsp",
        )

    def test_rejects_odd_evaluation_count(self):
        with self.assertRaises(ValueError):
            PoolTrainConfig(
                strategy="uniform",
                seed=1,
                budget_seconds=60,
                evaluation_games=99,
            )

    def test_rejects_invalid_integer_settings(self):
        invalid_settings = [
            {"seed": True},
            {"seed": -1},
            {"evaluation_games": 100.5},
            {"max_pool_size": False},
            {"log_every": 2.5},
            {"max_episodes": 1.5},
        ]

        for settings in invalid_settings:
            with self.subTest(settings=settings):
                arguments = {
                    "strategy": "uncertainty",
                    "seed": 1,
                    "budget_seconds": 60,
                }
                arguments.update(settings)
                with self.assertRaises(ValueError):
                    PoolTrainConfig(**arguments)

    def test_rejects_checkpoint_interval_over_budget(self):
        with self.assertRaises(ValueError):
            PoolTrainConfig(
                strategy="latest",
                seed=1,
                budget_seconds=10,
                checkpoint_seconds=20,
            )

    def test_rejects_non_finite_timing(self):
        invalid_timings = [
            {"budget_seconds": float("nan")},
            {"budget_seconds": float("inf")},
            {
                "budget_seconds": 60,
                "checkpoint_seconds": float("nan"),
            },
        ]

        for timing in invalid_timings:
            with self.subTest(timing=timing):
                with self.assertRaises(ValueError):
                    PoolTrainConfig(
                        strategy="latest",
                        seed=1,
                        **timing,
                    )

    def test_rejects_unsafe_run_name(self):
        with self.assertRaises(ValueError):
            PoolTrainConfig(
                strategy="uncertainty",
                seed=1,
                budget_seconds=60,
                run_name="../outside-project",
            )


if __name__ == "__main__":
    unittest.main()
