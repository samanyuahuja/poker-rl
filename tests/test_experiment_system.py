"""Tests for the reproducible experiment interface."""
import csv
import tempfile

from poker_rl.benchmark import read_training_manifest
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

class BenchmarkManifestTests(unittest.TestCase):
    def write_manifest(self, directory, rows):
        manifest_path = Path(directory) / "manifest.csv"

        with manifest_path.open(
            "w",
            newline="",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=[
                    "algorithm",
                    "training_seed",
                    "budget_seconds",
                    "run_directory",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        return manifest_path

    def test_accepts_matched_training_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [
                {
                    "algorithm": algorithm,
                    "training_seed": 1,
                    "budget_seconds": 300,
                    "run_directory": (
                        Path(directory) / algorithm
                    ),
                }
                for algorithm in ("dqn", "nfsp", "cfr")
            ]

            manifest_path = self.write_manifest(
                directory,
                rows,
            )

            runs, seeds, budget = read_training_manifest(
                manifest_path
            )

            self.assertEqual(seeds, [1])
            self.assertEqual(budget, 300)
            self.assertEqual(len(runs), 3)

    def test_rejects_unequal_training_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [
                {
                    "algorithm": "dqn",
                    "training_seed": 1,
                    "budget_seconds": 300,
                    "run_directory": (
                        Path(directory) / "dqn"
                    ),
                },
                {
                    "algorithm": "nfsp",
                    "training_seed": 1,
                    "budget_seconds": 300,
                    "run_directory": (
                        Path(directory) / "nfsp"
                    ),
                },
                {
                    "algorithm": "cfr",
                    "training_seed": 1,
                    "budget_seconds": 600,
                    "run_directory": (
                        Path(directory) / "cfr"
                    ),
                },
            ]

            manifest_path = self.write_manifest(
                directory,
                rows,
            )

            with self.assertRaises(ValueError):
                read_training_manifest(manifest_path)

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
    def test_benchmark_command(self):
        args = self.parser.parse_args(
            [
                "benchmark",
                "--manifest",
                "artifacts/suite.csv",
                "--evaluation-seeds",
                "11",
                "22",
                "--games-per-seat",
                "1000",
                "--raw-output",
                "results/raw.csv",
                "--summary-output",
                "results/summary.csv",
            ]
        )

        self.assertEqual(args.command, "benchmark")
        self.assertEqual(
            args.manifest,
            Path("artifacts/suite.csv"),
        )
        self.assertEqual(
            args.evaluation_seeds,
            [11, 22],
        )
        self.assertEqual(args.games_per_seat, 1000)

    def test_pool_train_command(self):
        args = self.parser.parse_args(
            [
                "pool-train",
                "--strategy",
                "uncertainty",
                "--seed",
                "9",
                "--budget-seconds",
                "300",
                "--checkpoint-seconds",
                "30",
                "--evaluation-games",
                "100",
                "--max-pool-size",
                "10",
            ]
        )

        self.assertEqual(args.command, "pool-train")
        self.assertEqual(
            args.strategy,
            "uncertainty",
        )
        self.assertEqual(args.seed, 9)
        self.assertEqual(args.budget_seconds, 300)
        self.assertEqual(args.checkpoint_seconds, 30)
        self.assertEqual(args.evaluation_games, 100)
        self.assertEqual(args.max_pool_size, 10)
    def test_pool_suite_command(self):
        args = self.parser.parse_args(
            [
                "pool-suite",
                "--strategies",
                "latest",
                "uniform",
                "pfsp",
                "uncertainty",
                "--seeds",
                "1",
                "2",
                "--budget-seconds",
                "300",
                "--checkpoint-seconds",
                "30",
                "--evaluation-games",
                "100",
                "--max-pool-size",
                "10",
                "--output",
                "results/pool_suite.csv",
            ]
        )

        self.assertEqual(
            args.command,
            "pool-suite",
        )
        self.assertEqual(
            args.strategies,
            [
                "latest",
                "uniform",
                "pfsp",
                "uncertainty",
            ],
        )
        self.assertEqual(args.seeds, [1, 2])
        self.assertEqual(
            args.budget_seconds,
            300,
        )
        self.assertEqual(
            args.checkpoint_seconds,
            30,
        )
        self.assertEqual(
            args.evaluation_games,
            100,
        )
        self.assertEqual(
            args.max_pool_size,
            10,
        )
        self.assertEqual(
            args.output,
            Path("results/pool_suite.csv"),
        )

    def test_pool_benchmark_command(self):
        args = self.parser.parse_args(
            [
                "pool-benchmark",
                "--pool-manifest",
                "artifacts/pool.csv",
                "--baseline-manifest",
                "results/baseline.csv",
                "--evaluation-seeds",
                "11",
                "22",
                "--games-per-seat",
                "2000",
                "--raw-output",
                "results/pool_raw.csv",
                "--summary-output",
                "results/pool_summary.csv",
                "--aggregate-output",
                "results/pool_aggregate.csv",
            ]
        )

        self.assertEqual(
            args.command,
            "pool-benchmark",
        )
        self.assertEqual(
            args.pool_manifest,
            Path("artifacts/pool.csv"),
        )
        self.assertEqual(
            args.baseline_manifest,
            Path("results/baseline.csv"),
        )
        self.assertEqual(
            args.evaluation_seeds,
            [11, 22],
        )
        self.assertEqual(
            args.games_per_seat,
            2000,
        )
        self.assertEqual(
            args.raw_output,
            Path("results/pool_raw.csv"),
        )
        self.assertEqual(
            args.summary_output,
            Path("results/pool_summary.csv"),
        )
        self.assertEqual(
            args.aggregate_output,
            Path("results/pool_aggregate.csv"),
        )

    def test_pool_analyze_command(self):
        args = self.parser.parse_args(
            [
                "pool-analyze",
                "--raw-input",
                "results/pool_benchmark_raw.csv",
                "--output",
                "results/pool_benchmark_paired.csv",
            ]
        )

        self.assertEqual(
            args.command,
            "pool-analyze",
        )
        self.assertEqual(
            args.raw_input,
            Path(
                "results/"
                "pool_benchmark_raw.csv"
            ),
        )
        self.assertEqual(
            args.output,
            Path(
                "results/"
                "pool_benchmark_paired.csv"
            ),
        )

if __name__ == "__main__":
    unittest.main()
