"""Paired statistical analysis for controlled ablation results."""

from __future__ import annotations

import csv
import itertools
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from poker_rl.ablation_suite import ABLATION_BY_LABEL
from poker_rl.evaluation import confidence_interval_95
from poker_rl.pool_analysis import read_reference_seed_scores
from poker_rl.pool_benchmark import REFERENCE_AGENTS, write_rows


ReferenceScoreKey = Tuple[str, int, str]

CONTROL_STRATEGIES = {
    "beta-1": "uncertainty",
    "latest": "latest",
}

COMPARISON_FAMILIES = {
    "uncertainty_strength": (
        "beta-0",
        "beta-1",
        "beta-2",
    ),
    "pool_size": (
        "pool-5",
        "pool-10",
        "pool-20",
    ),
    "historical_selection": (
        "latest",
        "beta-1",
    ),
}
def read_ablation_reference_seed_scores(
    raw_input_path: Path,
) -> Tuple[
    Dict[ReferenceScoreKey, float],
    List[int],
    List[int],
]:
    if not raw_input_path.exists():
        raise FileNotFoundError(
            "Raw ablation results do not exist: "
            f"{raw_input_path}"
        )

    grouped_scores = defaultdict(list)
    evaluation_seed_sets: Dict[
        ReferenceScoreKey,
        Set[int],
    ] = defaultdict(set)
    observed_rows = set()

    with raw_input_path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)

        required_fields = {
            "ablation_label",
            "training_seed",
            "evaluation_seed",
            "reference_agent",
            "combined_payoff",
        }

        if reader.fieldnames is None:
            raise ValueError(
                "Raw ablation file has no header."
            )

        missing_fields = (
            required_fields - set(reader.fieldnames)
        )

        if missing_fields:
            raise ValueError(
                "Raw ablation file is missing fields: "
                + ", ".join(sorted(missing_fields))
            )

        for row in reader:
            label = row["ablation_label"]
            reference = row["reference_agent"]
            training_seed = int(
                row["training_seed"]
            )
            evaluation_seed = int(
                row["evaluation_seed"]
            )
            payoff = float(row["combined_payoff"])

            if label not in ABLATION_BY_LABEL:
                raise ValueError(
                    f"Unknown ablation label: {label}"
                )

            if reference not in REFERENCE_AGENTS:
                raise ValueError(
                    "Unknown reference agent: "
                    f"{reference}"
                )

            row_key = (
                label,
                reference,
                training_seed,
                evaluation_seed,
            )

            if row_key in observed_rows:
                raise ValueError(
                    "Duplicate ablation observation "
                    f"for {row_key}."
                )

            observed_rows.add(row_key)

            score_key = (
                label,
                training_seed,
                reference,
            )
            grouped_scores[score_key].append(payoff)
            evaluation_seed_sets[
                score_key
            ].add(evaluation_seed)

    if not grouped_scores:
        raise ValueError(
            "No ablation observations were found."
        )

    training_seeds = sorted(
        {
            training_seed
            for _, training_seed, _ in grouped_scores
        }
    )

    expected_keys = {
        (label, training_seed, reference)
        for label in ABLATION_BY_LABEL
        for training_seed in training_seeds
        for reference in REFERENCE_AGENTS
    }
    actual_keys = set(grouped_scores)

    if actual_keys != expected_keys:
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys

        raise ValueError(
            "Ablation benchmark coverage is incomplete. "
            f"Missing={sorted(missing)}, "
            f"extra={sorted(extra)}."
        )

    first_key = next(iter(evaluation_seed_sets))
    expected_evaluation_seeds = (
        evaluation_seed_sets[first_key]
    )

    for key, seeds in evaluation_seed_sets.items():
        if seeds != expected_evaluation_seeds:
            raise ValueError(
                "Every ablation matchup must use "
                "the same evaluation seeds. "
                f"{key} has {sorted(seeds)}, "
                f"expected "
                f"{sorted(expected_evaluation_seeds)}."
            )

    seed_scores = {
        key: statistics.mean(values)
        for key, values in grouped_scores.items()
    }

    return (
        seed_scores,
        training_seeds,
        sorted(expected_evaluation_seeds),
    )

def build_combined_metric_values(
    ablation_raw_input_path: Path,
    primary_raw_input_path: Path,
) -> Tuple[
    Dict[Tuple[str, str], List[float]],
    List[int],
    List[int],
]:
    (
        ablation_scores,
        ablation_training_seeds,
        ablation_evaluation_seeds,
    ) = read_ablation_reference_seed_scores(
        ablation_raw_input_path
    )

    (
        primary_scores,
        primary_training_seeds,
        primary_evaluation_seeds,
    ) = read_reference_seed_scores(
        primary_raw_input_path
    )

    if (
        ablation_training_seeds
        != primary_training_seeds
    ):
        raise ValueError(
            "Ablation and primary results must use "
            "the same training seeds."
        )

    if (
        ablation_evaluation_seeds
        != primary_evaluation_seeds
    ):
        raise ValueError(
            "Ablation and primary results must use "
            "the same evaluation seeds."
        )

    combined_scores = dict(ablation_scores)

    for control_label, strategy in (
        CONTROL_STRATEGIES.items()
    ):
        for training_seed in (
            ablation_training_seeds
        ):
            for reference in REFERENCE_AGENTS:
                combined_scores[
                    (
                        control_label,
                        training_seed,
                        reference,
                    )
                ] = primary_scores[
                    (
                        strategy,
                        training_seed,
                        reference,
                    )
                ]

    configurations = (
        tuple(ABLATION_BY_LABEL)
        + tuple(CONTROL_STRATEGIES)
    )
    metric_values = {}

    for configuration in configurations:
        reference_mean_values = []
        worst_case_values = []

        for training_seed in (
            ablation_training_seeds
        ):
            scores = [
                combined_scores[
                    (
                        configuration,
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
                configuration,
                "reference_mean_payoff",
            )
        ] = reference_mean_values

        metric_values[
            (
                configuration,
                "reference_worst_case_payoff",
            )
        ] = worst_case_values

    return (
        metric_values,
        ablation_training_seeds,
        ablation_evaluation_seeds,
    )

def run_ablation_paired_analysis(
    ablation_raw_input_path: Path,
    primary_raw_input_path: Path,
    output_path: Path,
) -> List[dict]:
    (
        metric_values,
        training_seeds,
        evaluation_seeds,
    ) = build_combined_metric_values(
        ablation_raw_input_path=(
            ablation_raw_input_path
        ),
        primary_raw_input_path=(
            primary_raw_input_path
        ),
    )

    rows = []

    for family, configurations in (
        COMPARISON_FAMILIES.items()
    ):
        for metric in (
            "reference_mean_payoff",
            "reference_worst_case_payoff",
        ):
            for agent_a, agent_b in (
                itertools.combinations(
                    configurations,
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

                mean_difference = statistics.mean(
                    differences
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
                        conclusion = "agent_a_higher"
                    elif upper_bound < 0:
                        conclusion = "agent_b_higher"
                    else:
                        conclusion = "inconclusive"

                rows.append(
                    {
                        "family": family,
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
                    f"{family}, {metric}, "
                    f"{agent_a} minus {agent_b}: "
                    f"{mean_difference:.4f}"
                    f"{interval_text} "
                    f"({conclusion})"
                )

    write_rows(
        output_path=output_path,
        fieldnames=[
            "family",
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
        rows=rows,
    )

    print(
        "Saved ablation paired comparisons to "
        f"{output_path}"
    )

    return rows