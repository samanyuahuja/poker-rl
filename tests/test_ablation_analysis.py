"""Tests for paired controlled-ablation analysis."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import poker_rl.ablation_analysis as ablation_analysis
from poker_rl.ablation_analysis import (
    CONTROL_STRATEGIES,
    build_combined_metric_values,
    read_ablation_reference_seed_scores,
    run_ablation_paired_analysis,
)
from poker_rl.ablation_suite import ABLATION_BY_LABEL
from poker_rl.pool_benchmark import REFERENCE_AGENTS


class AblationAnalysisTests(unittest.TestCase):
    def create_raw_file(
        self,
        directory: Path,
        seeds,
        evaluation_seeds,
        omitted_key=None,
        duplicate_first_row=False,
    ):
        output_path = directory / "ablation_raw.csv"
        rows = []

        for label in ABLATION_BY_LABEL:
            for reference in REFERENCE_AGENTS:
                for training_seed in seeds:
                    for evaluation_seed in evaluation_seeds:
                        key = (
                            label,
                            reference,
                            training_seed,
                            evaluation_seed,
                        )

                        if key == omitted_key:
                            continue

                        rows.append(
                            {
                                "ablation_label": label,
                                "training_seed": training_seed,
                                "evaluation_seed": evaluation_seed,
                                "reference_agent": reference,
                                "combined_payoff": (
                                    training_seed
                                    + evaluation_seed / 100
                                ),
                            }
                        )

        if duplicate_first_row:
            rows.append(dict(rows[0]))

        with output_path.open(
            "w",
            newline="",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=[
                    "ablation_label",
                    "training_seed",
                    "evaluation_seed",
                    "reference_agent",
                    "combined_payoff",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        return output_path

    def test_reads_complete_reference_scores(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            input_path = self.create_raw_file(
                directory=directory,
                seeds=[1, 2],
                evaluation_seeds=[11, 22],
            )

            scores, training, evaluation = (
                read_ablation_reference_seed_scores(
                    input_path
                )
            )

            self.assertEqual(len(scores), 40)
            self.assertEqual(training, [1, 2])
            self.assertEqual(evaluation, [11, 22])

    def test_rejects_duplicate_observation(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            input_path = self.create_raw_file(
                directory=directory,
                seeds=[1],
                evaluation_seeds=[11],
                duplicate_first_row=True,
            )

            with self.assertRaises(ValueError):
                read_ablation_reference_seed_scores(
                    input_path
                )

    def test_rejects_incomplete_coverage(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            input_path = self.create_raw_file(
                directory=directory,
                seeds=[1],
                evaluation_seeds=[11],
                omitted_key=(
                    "pool-20",
                    "cfr",
                    1,
                    11,
                ),
            )

            with self.assertRaises(ValueError):
                read_ablation_reference_seed_scores(
                    input_path
                )

    def test_rejects_mismatched_training_seeds(self):
        with (
            mock.patch.object(
                ablation_analysis,
                "read_ablation_reference_seed_scores",
                return_value=({}, [1], [11]),
            ),
            mock.patch.object(
                ablation_analysis,
                "read_reference_seed_scores",
                return_value=({}, [2], [11]),
            ),
        ):
            with self.assertRaises(ValueError):
                build_combined_metric_values(
                    ablation_raw_input_path=Path(
                        "ablation.csv"
                    ),
                    primary_raw_input_path=Path(
                        "primary.csv"
                    ),
                )

    def test_writes_fourteen_paired_comparisons(self):
        configurations = (
            tuple(ABLATION_BY_LABEL)
            + tuple(CONTROL_STRATEGIES)
        )
        metric_values = {}

        for index, configuration in enumerate(
            configurations
        ):
            for metric in (
                "reference_mean_payoff",
                "reference_worst_case_payoff",
            ):
                metric_values[
                    (configuration, metric)
                ] = [
                    float(index),
                    float(index),
                ]

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            output_path = directory / "paired.csv"

            with mock.patch.object(
                ablation_analysis,
                "build_combined_metric_values",
                return_value=(
                    metric_values,
                    [1, 2],
                    [11, 22],
                ),
            ):
                rows = run_ablation_paired_analysis(
                    ablation_raw_input_path=Path(
                        "ablation.csv"
                    ),
                    primary_raw_input_path=Path(
                        "primary.csv"
                    ),
                    output_path=output_path,
                )

            self.assertEqual(len(rows), 14)
            self.assertTrue(output_path.exists())

            with output_path.open(
                newline="",
            ) as input_file:
                saved_rows = list(
                    csv.DictReader(input_file)
                )

            self.assertEqual(len(saved_rows), 14)


if __name__ == "__main__":
    unittest.main()