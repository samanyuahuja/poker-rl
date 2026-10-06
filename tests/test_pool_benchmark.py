"""Tests for opponent-pool benchmark evaluation."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import poker_rl.pool_benchmark as pool_benchmark
from poker_rl.pool_benchmark import (
    POOL_STRATEGIES,
    read_pool_training_manifest,
    run_pool_benchmark,
)


class PoolBenchmarkTests(unittest.TestCase):
    def create_pool_manifest(
        self,
        directory: Path,
        seeds,
        budget_seconds=300,
        omitted_strategy=None,
    ):
        manifest_path = (
            directory / "pool_manifest.csv"
        )
        rows = []

        for strategy in POOL_STRATEGIES:
            if strategy == omitted_strategy:
                continue

            for seed in seeds:
                run_directory = (
                    directory
                    / f"{strategy}-seed{seed}"
                )
                run_directory.mkdir()

                with (
                    run_directory / "summary.json"
                ).open("w") as output_file:
                    json.dump(
                        {"status": "complete"},
                        output_file,
                    )

                rows.append(
                    {
                        "strategy": strategy,
                        "training_seed": seed,
                        "budget_seconds": (
                            budget_seconds
                        ),
                        "run_directory": str(
                            run_directory
                        ),
                    }
                )

        with manifest_path.open(
            "w",
            newline="",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=[
                    "strategy",
                    "training_seed",
                    "budget_seconds",
                    "run_directory",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        return manifest_path

    def baseline_runs(
        self,
        directory: Path,
        seeds,
    ):
        return {
            (algorithm, seed): (
                directory
                / f"{algorithm}-seed{seed}"
            )
            for algorithm in (
                "dqn",
                "nfsp",
                "cfr",
            )
            for seed in seeds
        }

    def test_reads_complete_matched_manifest(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            manifest_path = (
                self.create_pool_manifest(
                    directory=directory,
                    seeds=[1, 2],
                )
            )

            runs, seeds, budget = (
                read_pool_training_manifest(
                    manifest_path
                )
            )

            self.assertEqual(
                len(runs),
                8,
            )
            self.assertEqual(
                seeds,
                [1, 2],
            )
            self.assertEqual(
                budget,
                300,
            )

    def test_rejects_missing_strategy(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            manifest_path = (
                self.create_pool_manifest(
                    directory=directory,
                    seeds=[1],
                    omitted_strategy="pfsp",
                )
            )

            with self.assertRaises(ValueError):
                read_pool_training_manifest(
                    manifest_path
                )

    def test_rejects_duplicate_evaluation_seeds(self):
        with self.assertRaises(ValueError):
            run_pool_benchmark(
                pool_manifest_path=Path(
                    "pool.csv"
                ),
                baseline_manifest_path=Path(
                    "baseline.csv"
                ),
                evaluation_seeds=[11, 11],
                games_per_seat=100,
                raw_output_path=Path(
                    "raw.csv"
                ),
                summary_output_path=Path(
                    "summary.csv"
                ),
                aggregate_output_path=Path(
                    "aggregate.csv"
                ),
            )

    def test_rejects_mismatched_budgets(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            manifest_path = (
                self.create_pool_manifest(
                    directory=directory,
                    seeds=[1],
                    budget_seconds=300,
                )
            )

            with mock.patch.object(
                pool_benchmark,
                "read_training_manifest",
                return_value=(
                    self.baseline_runs(
                        directory,
                        [1],
                    ),
                    [1],
                    600,
                ),
            ):
                with self.assertRaises(
                    ValueError
                ):
                    run_pool_benchmark(
                        pool_manifest_path=(
                            manifest_path
                        ),
                        baseline_manifest_path=(
                            directory
                            / "baseline.csv"
                        ),
                        evaluation_seeds=[11],
                        games_per_seat=10,
                        raw_output_path=(
                            directory / "raw.csv"
                        ),
                        summary_output_path=(
                            directory
                            / "summary.csv"
                        ),
                        aggregate_output_path=(
                            directory
                            / "aggregate.csv"
                        ),
                    )

    def test_writes_all_matchups_and_aggregates(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            manifest_path = (
                self.create_pool_manifest(
                    directory=directory,
                    seeds=[1],
                )
            )

            raw_output = (
                directory / "raw.csv"
            )
            summary_output = (
                directory / "summary.csv"
            )
            aggregate_output = (
                directory / "aggregate.csv"
            )

            with (
                mock.patch.object(
                    pool_benchmark,
                    "read_training_manifest",
                    return_value=(
                        self.baseline_runs(
                            directory,
                            [1],
                        ),
                        [1],
                        300,
                    ),
                ),
                mock.patch.object(
                    pool_benchmark,
                    "evaluate_both_seats",
                    return_value=(0.4, 0.6),
                ) as evaluate,
            ):
                (
                    summary_rows,
                    aggregate_rows,
                ) = run_pool_benchmark(
                    pool_manifest_path=(
                        manifest_path
                    ),
                    baseline_manifest_path=(
                        directory
                        / "baseline.csv"
                    ),
                    evaluation_seeds=[11],
                    games_per_seat=10,
                    raw_output_path=raw_output,
                    summary_output_path=(
                        summary_output
                    ),
                    aggregate_output_path=(
                        aggregate_output
                    ),
                )

            self.assertEqual(
                len(summary_rows),
                22,
            )
            self.assertEqual(
                len(aggregate_rows),
                8,
            )
            self.assertEqual(
                evaluate.call_count,
                22,
            )
            self.assertTrue(
                raw_output.exists()
            )
            self.assertTrue(
                summary_output.exists()
            )
            self.assertTrue(
                aggregate_output.exists()
            )

            worst_case_rows = [
                row
                for row in aggregate_rows
                if row["metric"] == (
                    "reference_"
                    "worst_case_payoff"
                )
            ]

            self.assertEqual(
                len(worst_case_rows),
                4,
            )

            for row in worst_case_rows:
                self.assertAlmostEqual(
                    row["mean_payoff"],
                    0.5,
                )


if __name__ == "__main__":
    unittest.main()