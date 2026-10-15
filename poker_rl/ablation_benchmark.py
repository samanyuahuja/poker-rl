"""Fixed-reference evaluation for opponent-pool ablations."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Union

from poker_rl.ablation_suite import (
    ABLATION_BY_LABEL,
)
from poker_rl.benchmark import (
    evaluate_both_seats,
    read_training_manifest,
)
from poker_rl.config import PROJECT_ROOT
from poker_rl.evaluation import (
    confidence_interval_95,
)
from poker_rl.pool_benchmark import (
    REFERENCE_AGENTS,
    write_rows,
)

AgentSpec = Union[str, Path]


def read_json(path: Path) -> dict:
    with path.open() as input_file:
        return json.load(input_file)


def read_ablation_manifest(
    manifest_path: Path,
) -> Tuple[
    Dict[Tuple[str, int], Path],
    List[int],
    float,
]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Ablation manifest does not exist: "
            f"{manifest_path}"
        )

    runs: Dict[Tuple[str, int], Path] = {}
    budgets = set()

    with manifest_path.open(
        newline="",
    ) as input_file:
        reader = csv.DictReader(input_file)

        required_fields = {
            "ablation_label",
            "training_seed",
            "budget_seconds",
            "run_directory",
        }

        if reader.fieldnames is None:
            raise ValueError(
                "Ablation manifest has no header."
            )

        missing_fields = (
            required_fields - set(reader.fieldnames)
        )

        if missing_fields:
            raise ValueError(
                "Ablation manifest is missing fields: "
                + ", ".join(
                    sorted(missing_fields)
                )
            )

        for row in reader:
            label = row["ablation_label"]
            training_seed = int(
                row["training_seed"]
            )
            budget_seconds = float(
                row["budget_seconds"]
            )

            if label not in ABLATION_BY_LABEL:
                raise ValueError(
                    f"Unknown ablation label: "
                    f"{label}"
                )

            run_directory = Path(
                row["run_directory"]
            )

            if not run_directory.is_absolute():
                run_directory = (
                    PROJECT_ROOT / run_directory
                )

            run_directory = (
                run_directory.resolve()
            )

            key = (label, training_seed)

            if key in runs:
                raise ValueError(
                    "Duplicate ablation run for "
                    f"{label}, seed "
                    f"{training_seed}."
                )

            summary_path = (
                run_directory / "summary.json"
            )

            if not summary_path.exists():
                raise ValueError(
                    f"Missing run summary: "
                    f"{summary_path}"
                )

            summary = read_json(summary_path)

            if summary.get("status") != "complete":
                raise ValueError(
                    f"Ablation run is not complete: "
                    f"{run_directory}"
                )

            runs[key] = run_directory
            budgets.add(budget_seconds)

    if len(budgets) != 1:
        raise ValueError(
            "All ablation runs must use one "
            "common training budget."
        )

    labels = tuple(
        ABLATION_BY_LABEL
    )

    label_seed_sets = {
        label: {
            seed
            for stored_label, seed in runs
            if stored_label == label
        }
        for label in labels
    }

    reference_seeds = label_seed_sets[
        labels[0]
    ]

    if not reference_seeds:
        raise ValueError(
            f"No {labels[0]} runs were found."
        )

    for label, seeds in (
        label_seed_sets.items()
    ):
        if seeds != reference_seeds:
            raise ValueError(
                "Every ablation must use the "
                "same training seeds. "
                f"{label} has {sorted(seeds)}, "
                f"expected "
                f"{sorted(reference_seeds)}."
            )

    return (
        runs,
        sorted(reference_seeds),
        budgets.pop(),
    )


def resolve_reference_agent(
    reference: str,
    training_seed: int,
    baseline_runs: Dict[
        Tuple[str, int],
        Path,
    ],
) -> AgentSpec:
    if reference == "random":
        return "random"

    return baseline_runs[
        (reference, training_seed)
    ]


def run_ablation_benchmark(
    ablation_manifest_path: Path,
    baseline_manifest_path: Path,
    evaluation_seeds: List[int],
    games_per_seat: int,
    raw_output_path: Path,
    summary_output_path: Path,
    aggregate_output_path: Path,
) -> Tuple[List[dict], List[dict]]:
    if not evaluation_seeds:
        raise ValueError(
            "At least one evaluation seed is "
            "required."
        )

    if len(evaluation_seeds) != len(
        set(evaluation_seeds)
    ):
        raise ValueError(
            "Evaluation seeds must not contain "
            "duplicates."
        )

    if games_per_seat <= 0:
        raise ValueError(
            "games_per_seat must be greater "
            "than zero."
        )

    (
        ablation_runs,
        ablation_training_seeds,
        ablation_budget,
    ) = read_ablation_manifest(
        ablation_manifest_path
    )

    (
        baseline_runs,
        baseline_training_seeds,
        baseline_budget,
    ) = read_training_manifest(
        baseline_manifest_path
    )

    if (
        ablation_training_seeds
        != baseline_training_seeds
    ):
        raise ValueError(
            "Ablation and baseline runs must "
            "use the same training seeds."
        )

    if ablation_budget != baseline_budget:
        raise ValueError(
            "Ablation and baseline runs must "
            "use the same training budget."
        )

    labels = tuple(
        ABLATION_BY_LABEL
    )

    raw_rows: List[dict] = []
    summary_rows: List[dict] = []

    reference_seed_scores: Dict[
        Tuple[str, int, str],
        float,
    ] = {}

    for label in labels:
        spec = ABLATION_BY_LABEL[label]

        for reference in REFERENCE_AGENTS:
            training_seed_means = []

            for training_seed in (
                ablation_training_seeds
            ):
                agent_spec = ablation_runs[
                    (label, training_seed)
                ]

                reference_spec = (
                    resolve_reference_agent(
                        reference=reference,
                        training_seed=(
                            training_seed
                        ),
                        baseline_runs=(
                            baseline_runs
                        ),
                    )
                )

                evaluation_scores = []

                for evaluation_seed in (
                    evaluation_seeds
                ):
                    (
                        seat_zero,
                        seat_one,
                    ) = evaluate_both_seats(
                        agent_a_spec=agent_spec,
                        agent_b_spec=(
                            reference_spec
                        ),
                        evaluation_seed=(
                            evaluation_seed
                        ),
                        games_per_seat=(
                            games_per_seat
                        ),
                    )

                    combined_payoff = (
                        seat_zero + seat_one
                    ) / 2

                    evaluation_scores.append(
                        combined_payoff
                    )

                    raw_rows.append(
                        {
                            "ablation_label": (
                                label
                            ),
                            "ablation_family": (
                                spec.family
                            ),
                            "training_seed": (
                                training_seed
                            ),
                            "evaluation_seed": (
                                evaluation_seed
                            ),
                            "training_budget_seconds": (
                                ablation_budget
                            ),
                            "reference_agent": (
                                reference
                            ),
                            "games_per_seat": (
                                games_per_seat
                            ),
                            "seat_0_payoff": (
                                seat_zero
                            ),
                            "seat_1_payoff": (
                                seat_one
                            ),
                            "combined_payoff": (
                                combined_payoff
                            ),
                        }
                    )

                training_seed_mean = (
                    statistics.mean(
                        evaluation_scores
                    )
                )
                training_seed_means.append(
                    training_seed_mean
                )

                reference_seed_scores[
                    (
                        label,
                        training_seed,
                        reference,
                    )
                ] = training_seed_mean

                print(
                    f"{label} vs {reference}, "
                    f"training seed "
                    f"{training_seed}: "
                    f"{training_seed_mean:.4f}"
                )

            mean_payoff = statistics.mean(
                training_seed_means
            )
            interval = confidence_interval_95(
                training_seed_means
            )

            summary_rows.append(
                {
                    "ablation_label": label,
                    "ablation_family": (
                        spec.family
                    ),
                    "reference_agent": (
                        reference
                    ),
                    "training_budget_seconds": (
                        ablation_budget
                    ),
                    "training_seed_count": len(
                        ablation_training_seeds
                    ),
                    "evaluation_seed_count": len(
                        evaluation_seeds
                    ),
                    "games_per_seat": (
                        games_per_seat
                    ),
                    "mean_payoff": mean_payoff,
                    "confidence_interval_95": (
                        ""
                        if interval is None
                        else interval
                    ),
                }
            )

            interval_text = (
                ""
                if interval is None
                else f" ± {interval:.4f}"
            )

            print(
                f"{label} vs {reference}: "
                f"{mean_payoff:.4f}"
                f"{interval_text}"
            )

    aggregate_rows: List[dict] = []

    for label in labels:
        spec = ABLATION_BY_LABEL[label]
        seed_reference_means = []
        seed_worst_case_payoffs = []

        for training_seed in (
            ablation_training_seeds
        ):
            scores = [
                reference_seed_scores[
                    (
                        label,
                        training_seed,
                        reference,
                    )
                ]
                for reference in (
                    REFERENCE_AGENTS
                )
            ]

            seed_reference_means.append(
                statistics.mean(scores)
            )
            seed_worst_case_payoffs.append(
                min(scores)
            )

        metric_values = {
            "reference_mean_payoff": (
                seed_reference_means
            ),
            "reference_worst_case_payoff": (
                seed_worst_case_payoffs
            ),
        }

        reference_summary_scores = {
            row["reference_agent"]: float(
                row["mean_payoff"]
            )
            for row in summary_rows
            if row["ablation_label"] == label
        }

        worst_opponent = min(
            reference_summary_scores,
            key=reference_summary_scores.get,
        )

        for metric, values in (
            metric_values.items()
        ):
            mean_value = statistics.mean(
                values
            )
            interval = confidence_interval_95(
                values
            )

            aggregate_rows.append(
                {
                    "ablation_label": label,
                    "ablation_family": (
                        spec.family
                    ),
                    "metric": metric,
                    "training_seed_count": len(
                        ablation_training_seeds
                    ),
                    "mean_payoff": mean_value,
                    "confidence_interval_95": (
                        ""
                        if interval is None
                        else interval
                    ),
                    "worst_opponent": (
                        worst_opponent
                        if metric
                        == (
                            "reference_"
                            "worst_case_payoff"
                        )
                        else ""
                    ),
                }
            )

            interval_text = (
                ""
                if interval is None
                else f" ± {interval:.4f}"
            )

            print(
                f"{label} {metric}: "
                f"{mean_value:.4f}"
                f"{interval_text}"
            )

    write_rows(
        output_path=raw_output_path,
        fieldnames=[
            "ablation_label",
            "ablation_family",
            "training_seed",
            "evaluation_seed",
            "training_budget_seconds",
            "reference_agent",
            "games_per_seat",
            "seat_0_payoff",
            "seat_1_payoff",
            "combined_payoff",
        ],
        rows=raw_rows,
    )

    write_rows(
        output_path=summary_output_path,
        fieldnames=[
            "ablation_label",
            "ablation_family",
            "reference_agent",
            "training_budget_seconds",
            "training_seed_count",
            "evaluation_seed_count",
            "games_per_seat",
            "mean_payoff",
            "confidence_interval_95",
        ],
        rows=summary_rows,
    )

    write_rows(
        output_path=aggregate_output_path,
        fieldnames=[
            "ablation_label",
            "ablation_family",
            "metric",
            "training_seed_count",
            "mean_payoff",
            "confidence_interval_95",
            "worst_opponent",
        ],
        rows=aggregate_rows,
    )

    print(
        f"Saved ablation raw results to "
        f"{raw_output_path}"
    )
    print(
        f"Saved ablation summary to "
        f"{summary_output_path}"
    )
    print(
        f"Saved ablation aggregates to "
        f"{aggregate_output_path}"
    )

    return summary_rows, aggregate_rows