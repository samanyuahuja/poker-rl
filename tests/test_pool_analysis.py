"""Tests for paired opponent-pool analysis."""

import csv
import tempfile
import unittest
from pathlib import Path

from poker_rl.pool_analysis import (
    read_reference_seed_scores,
    run_paired_analysis,
)
from poker_rl.pool_benchmark import (
    POOL_STRATEGIES,
    REFERENCE_AGENTS,
)


class PoolAnalysisTests(unittest.TestCase):
    def write_raw_results(
        self,
        path: Path,
        duplicate_first_row=False,
        omitted_group=None,
    ) -> None:
        strategy_values = {
            "latest": 0.0,
            "uniform": 1.0,
            "pfsp": 2.0,
            "uncertainty": 3.0,
        }

        reference_values = {
            "random": 0.3,
            "dqn": 0.1,
            "nfsp": 0.2,
            "cfr": 0.0,
        }

        rows = []

        for strategy in POOL_STRATEGIES:
            for training_seed in [
                1,
                2,
                3,
                4,
                5,
            ]:
                for reference in (
                    REFERENCE_AGENTS
                ):
                    group = (
                        strategy,
                        training_seed,
                        reference,
                    )

                    if group == omitted_group:
                        continue

                    for evaluation_seed in [
                        11,
                        22,
                    ]:
                        rows.append(
                            {
                                "matchup_group": (
                                    "reference"
                                ),
                                "training_seed": (
                                    training_seed
                                ),
                                "evaluation_seed": (
                                    evaluation_seed
                                ),
                                "agent_a": strategy,
                                "agent_b": reference,
                                "combined_payoff": (
                                    strategy_values[
                                        strategy
                                    ]
                                    + reference_values[
                                        reference
                                    ]
                                ),
                            }
                        )

        if duplicate_first_row:
            rows.append(dict(rows[0]))

        with path.open(
            "w",
            newline="",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=[
                    "matchup_group",
                    "training_seed",
                    "evaluation_seed",
                    "agent_a",
                    "agent_b",
                    "combined_payoff",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_reads_complete_reference_scores(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "raw.csv"

            self.write_raw_results(path)

            (
                scores,
                training_seeds,
                evaluation_seeds,
            ) = read_reference_seed_scores(
                path
            )

            self.assertEqual(
                len(scores),
                80,
            )
            self.assertEqual(
                training_seeds,
                [1, 2, 3, 4, 5],
            )
            self.assertEqual(
                evaluation_seeds,
                [11, 22],
            )

    def test_rejects_duplicate_observation(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "raw.csv"

            self.write_raw_results(
                path,
                duplicate_first_row=True,
            )

            with self.assertRaises(ValueError):
                read_reference_seed_scores(
                    path
                )

    def test_rejects_incomplete_coverage(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "raw.csv"

            self.write_raw_results(
                path,
                omitted_group=(
                    "uncertainty",
                    5,
                    "cfr",
                ),
            )

            with self.assertRaises(ValueError):
                read_reference_seed_scores(
                    path
                )

    def test_writes_twelve_paired_comparisons(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            input_path = (
                directory / "raw.csv"
            )
            output_path = (
                directory / "paired.csv"
            )

            self.write_raw_results(
                input_path
            )

            rows = run_paired_analysis(
                raw_input_path=input_path,
                output_path=output_path,
            )

            self.assertEqual(
                len(rows),
                12,
            )
            self.assertTrue(
                output_path.exists()
            )

            for row in rows:
                self.assertEqual(
                    row["conclusion"],
                    "agent_b_higher",
                )
                self.assertLess(
                    row["mean_difference"],
                    0,
                )


if __name__ == "__main__":
    unittest.main()