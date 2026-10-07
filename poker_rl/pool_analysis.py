"""Paired statistical analysis for opponent-pool benchmark results."""

from __future__ import annotations

import csv
import itertools
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from poker_rl.evaluation import confidence_interval_95
from poker_rl.pool_benchmark import (
    POOL_STRATEGIES,
    REFERENCE_AGENTS,
)


ReferenceScoreKey = Tuple[str, int, str]


def read_reference_seed_scores(
    raw_input_path: Path,
) -> Tuple[
    Dict[ReferenceScoreKey, float],
    List[int],
    List[int],
]:
    if not raw_input_path.exists():
        raise FileNotFoundError(
            f"Raw benchmark results do not exist: "
            f"{raw_input_path}"
        )

    grouped_scores = defaultdict(list)
    evaluation_seed_sets: Dict[
        ReferenceScoreKey,
        Set[int],
    ] = defaultdict(set)
    observed_rows = set()

    with raw_input_path.open(
        newline="",
    ) as input_file:
        reader = csv.DictReader(input_file)

        required_fields = {
            "matchup_group",
            "training_seed",
            "evaluation_seed",
            "agent_a",
            "agent_b",
            "combined_payoff",
        }

        if reader.fieldnames is None:
            raise ValueError(
                "Raw benchmark file has no header."
            )

        missing_fields = (
            required_fields - set(reader.fieldnames)
        )

        if missing_fields:
            raise ValueError(
                "Raw benchmark file is missing fields: "
                + ", ".join(
                    sorted(missing_fields)
                )
            )

        for row in reader:
            if row["matchup_group"] != "reference":
                continue

            strategy = row["agent_a"]
            reference = row["agent_b"]
            training_seed = int(
                row["training_seed"]
            )
            evaluation_seed = int(
                row["evaluation_seed"]
            )
            payoff = float(
                row["combined_payoff"]
            )

            if strategy not in POOL_STRATEGIES:
                raise ValueError(
                    f"Unknown strategy: {strategy}"
                )

            if reference not in REFERENCE_AGENTS:
                raise ValueError(
                    f"Unknown reference agent: "
                    f"{reference}"
                )

            row_key = (
                strategy,
                reference,
                training_seed,
                evaluation_seed,
            )

            if row_key in observed_rows:
                raise ValueError(
                    "Duplicate reference observation "
                    f"for {row_key}."
                )

            observed_rows.add(row_key)

            score_key = (
                strategy,
                training_seed,
                reference,
            )

            grouped_scores[score_key].append(
                payoff
            )
            evaluation_seed_sets[
                score_key
            ].add(evaluation_seed)

    if not grouped_scores:
        raise ValueError(
            "No reference-matchup rows were found."
        )

    training_seeds = sorted(
        {
            training_seed
            for (
                _,
                training_seed,
                _,
            ) in grouped_scores
        }
    )

    expected_keys = {
        (
            strategy,
            training_seed,
            reference,
        )
        for strategy in POOL_STRATEGIES
        for training_seed in training_seeds
        for reference in REFERENCE_AGENTS
    }

    actual_keys = set(grouped_scores)

    if actual_keys != expected_keys:
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys

        raise ValueError(
            "Reference benchmark coverage is "
            "incomplete. "
            f"Missing={sorted(missing)}, "
            f"extra={sorted(extra)}."
        )

    first_key = next(
        iter(evaluation_seed_sets)
    )
    expected_evaluation_seeds = (
        evaluation_seed_sets[first_key]
    )

    for key, seeds in (
        evaluation_seed_sets.items()
    ):
        if seeds != expected_evaluation_seeds:
            raise ValueError(
                "Every reference matchup must use "
                "the same evaluation seeds. "
                f"{key} has {sorted(seeds)}, "
                f"expected "
                f"{sorted(expected_evaluation_seeds)}."
            )

    seed_scores = {
        key: statistics.mean(values)
        for key, values in (
            grouped_scores.items()
        )
    }

    return (
        seed_scores,
        training_seeds,
        sorted(expected_evaluation_seeds),
    )


def build_metric_values(
    seed_scores: Dict[
        ReferenceScoreKey,
        float,
    ],
    training_seeds: List[int],
) -> Dict[Tuple[str, str], List[float]]:
    metric_values = {}

    for strategy in POOL_STRATEGIES:
        reference_mean_values = []
        worst_case_values = []

        for training_seed in training_seeds:
            scores = [
                seed_scores[
                    (
                        strategy,
                        training_seed,
                        reference,
                    )
                ]
                for reference in REFERENCE_AGENTS
            ]

            reference_mean_values.append(
                statistics.mean(scores)
            )
            worst_case_values.append(
                min(scores)
            )

        metric_values[
            (
                strategy,
                "reference_mean_payoff",
            )
        ] = reference_mean_values

        metric_values[
            (
                strategy,
                "reference_worst_case_payoff",
            )
        ] = worst_case_values

    return metric_values


def run_paired_analysis(
    raw_input_path: Path,
    output_path: Path,
) -> List[dict]:
    (
        seed_scores,
        training_seeds,
        evaluation_seeds,
    ) = read_reference_seed_scores(
        raw_input_path
    )

    metric_values = build_metric_values(
        seed_scores=seed_scores,
        training_seeds=training_seeds,
    )

    rows = []

    for metric in (
        "reference_mean_payoff",
        "reference_worst_case_payoff",
    ):
        for agent_a, agent_b in (
            itertools.combinations(
                POOL_STRATEGIES,
                2,
            )
        ):
            differences = [
                value_a - value_b
                for value_a, value_b in zip(
                    metric_values[
                        (agent_a, metric)
                    ],
                    metric_values[
                        (agent_b, metric)
                    ],
                )
            ]

            mean_difference = (
                statistics.mean(differences)
            )
            interval = confidence_interval_95(
                differences
            )

            if interval is None:
                lower_bound = None
                upper_bound = None
                conclusion = "inconclusive"
            else:
                lower_bound = (
                    mean_difference - interval
                )
                upper_bound = (
                    mean_difference + interval
                )

                if lower_bound > 0:
                    conclusion = (
                        "agent_a_higher"
                    )
                elif upper_bound < 0:
                    conclusion = (
                        "agent_b_higher"
                    )
                else:
                    conclusion = "inconclusive"

            rows.append(
                {
                    "metric": metric,
                    "agent_a": agent_a,
                    "agent_b": agent_b,
                    "difference_definition": (
                        "agent_a_minus_agent_b"
                    ),
                    "training_seed_count": len(
                        training_seeds
                    ),
                    "evaluation_seed_count": len(
                        evaluation_seeds
                    ),
                    "mean_difference": (
                        mean_difference
                    ),
                    "confidence_interval_95": (
                        ""
                        if interval is None
                        else interval
                    ),
                    "lower_bound_95": (
                        ""
                        if lower_bound is None
                        else lower_bound
                    ),
                    "upper_bound_95": (
                        ""
                        if upper_bound is None
                        else upper_bound
                    ),
                    "conclusion": conclusion,
                }
            )

            interval_text = (
                ""
                if interval is None
                else f" ± {interval:.4f}"
            )

            print(
                f"{metric}, {agent_a} minus "
                f"{agent_b}: "
                f"{mean_difference:.4f}"
                f"{interval_text} "
                f"({conclusion})"
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "metric",
                "agent_a",
                "agent_b",
                "difference_definition",
                "training_seed_count",
                "evaluation_seed_count",
                "mean_difference",
                "confidence_interval_95",
                "lower_bound_95",
                "upper_bound_95",
                "conclusion",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Saved paired comparisons to "
        f"{output_path}"
    )

    return rows