"""Tests for the reproducible experiment interface."""

import unittest
from pathlib import Path

from poker_rl.cli import build_parser
from poker_rl.config import TrainConfig
from poker_rl.evaluation import confidence_interval_95


class TrainConfigTests(unittest.TestCase):
    def test_valid_configuration(self):
        config = TrainConfig(
            algorithm="dqn",
            seed=42,
            budget_seconds=60,
        )

        self.assertEqual(config.algorithm, "dqn")
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.budget_seconds, 60)

    def test_rejects_unknown_algorithm(self):
        with self.assertRaises(ValueError):
            TrainConfig(
                algorithm="unknown",
                seed=42,
                budget_seconds=60,
            )

    def test_rejects_nonpositive_budget(self):
        with self.assertRaises(ValueError):
            TrainConfig(
                algorithm="dqn",
                seed=42,
                budget_seconds=0,
            )

    def test_rejects_nested_run_name(self):
        with self.assertRaises(ValueError):
            TrainConfig(
                algorithm="dqn",
                seed=42,
                budget_seconds=60,
                run_name="../outside-project",
            )


class ConfidenceIntervalTests(unittest.TestCase):
    def test_single_value_has_no_interval(self):
        self.assertIsNone(
            confidence_interval_95([1.0])
        )

    def test_five_value_interval(self):
        interval = confidence_interval_95(
            [1.0, 2.0, 3.0, 4.0, 5.0]
        )

        self.assertAlmostEqual(
            interval,
            1.963,
            places=3,
        )


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def test_train_command(self):
        args = self.parser.parse_args(
            [
                "train",
                "--algorithm",
                "nfsp",
                "--seed",
                "7",
                "--budget-seconds",
                "30",
            ]
        )

        self.assertEqual(args.command, "train")
        self.assertEqual(args.algorithm, "nfsp")
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.budget_seconds, 30)

    def test_evaluate_command(self):
        args = self.parser.parse_args(
            [
                "evaluate",
                "--agent-a",
                "artifacts/agent-a",
                "--agent-b",
                "random",
                "--output",
                "results/test.csv",
            ]
        )

        self.assertEqual(args.command, "evaluate")
        self.assertEqual(
            args.agent_a,
            "artifacts/agent-a",
        )
        self.assertEqual(args.agent_b, "random")
        self.assertEqual(
            args.output,
            Path("results/test.csv"),
        )
        self.assertEqual(
            args.seeds,
            [11, 22, 33, 44, 55],
        )
        
    def test_suite_command(self):
        args = self.parser.parse_args(
            [
                "suite",
                "--algorithms",
                "dqn",
                "nfsp",
                "cfr",
                "--seeds",
                "1",
                "2",
                "3",
                "--budget-seconds",
                "300",
                "--output",
                "results/training_manifest.csv",
            ]
        )

        self.assertEqual(args.command, "suite")
        self.assertEqual(
            args.algorithms,
            ["dqn", "nfsp", "cfr"],
        )
        self.assertEqual(args.seeds, [1, 2, 3])
        self.assertEqual(args.budget_seconds, 300)
        self.assertEqual(
            args.output,
            Path("results/training_manifest.csv"),
        )


if __name__ == "__main__":
    unittest.main()