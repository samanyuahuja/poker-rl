"""Rigorous evaluation for historical opponent-pool strategies."""

from __future__ import annotations

import csv
import itertools
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Union

from poker_rl.benchmark import (
    evaluate_both_seats,
    read_training_manifest,
)
from poker_rl.config import PROJECT_ROOT
from poker_rl.evaluation import confidence_interval_95

AgentSpec = Union[str, Path]

POOL_STRATEGIES = (
    "latest",
    "uniform",
    "pfsp",
    "uncertainty",
)

REFERENCE_AGENTS = (
    "random",
    "dqn",
    "nfsp",
    "cfr",
)


def read_json(path: Path) -> dict:
    with path.open() as input_file:
        return json.load(input_file)


def read_pool_training_manifest(
    manifest_path: Path,
) -> Tuple[
    Dict[Tuple[str, int], Path],
    List[int],
    float,
]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Pool manifest does not exist: "
            f"{manifest_path}"
        )

    runs: Dict[Tuple[str, int], Path] = {}
    budgets = set()

    with manifest_path.open(
        newline="",
    ) as input_file:
        reader = csv.DictReader(input_file)

        required_fields = {
            "strategy",
            "training_seed",
            "budget_seconds",
            "run_directory",
        }

        if reader.fieldnames is None:
            raise ValueError(
                "Pool manifest has no header."
            )

        missing_fields = (
            required_fields - set(reader.fieldnames)
        )

        if missing_fields:
            raise ValueError(
                "Pool manifest is missing fields: "
                + ", ".join(
                    sorted(missing_fields)
                )
            )

        for row in reader:
            strategy = row["strategy"]
            training_seed = int(
                row["training_seed"]
            )
            budget_seconds = float(
                row["budget_seconds"]
            )

            if strategy not in POOL_STRATEGIES:
                raise ValueError(
                    f"Unknown pool strategy: "
                    f"{strategy}"
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

            key = (strategy, training_seed)

            if key in runs:
                raise ValueError(
                    "Duplicate pool run for "
                    f"{strategy}, seed "
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
                    f"Pool run is not complete: "
                    f"{run_directory}"
                )

            runs[key] = run_directory
            budgets.add(budget_seconds)

    if len(budgets) != 1:
        raise ValueError(
            "All pool runs must use one common "
            "training budget."
        )

    strategy_seed_sets = {
        strategy: {
            seed
            for stored_strategy, seed in runs
            if stored_strategy == strategy
        }
        for strategy in POOL_STRATEGIES
    }

    reference_seeds = strategy_seed_sets[
        POOL_STRATEGIES[0]
    ]

    if not reference_seeds:
        raise ValueError(
            "No latest-strategy runs were found."
        )

    for strategy, seeds in (
        strategy_seed_sets.items()
    ):
        if seeds != reference_seeds:
            raise ValueError(
                "Every strategy must use the same "
                "training seeds. "
                f"{strategy} has {sorted(seeds)}, "
                f"while latest has "
                f"{sorted(reference_seeds)}."
            )

    return (
        runs,
        sorted(reference_seeds),
        budgets.pop(),
    )


def resolve_agent(
    name: str,
    training_seed: int,
    pool_runs: Dict[
        Tuple[str, int],
        Path,
    ],
    baseline_runs: Dict[
        Tuple[str, int],
        Path,
    ],
) -> AgentSpec:
    if name == "random":
        return "random"

    if name in POOL_STRATEGIES:
        return pool_runs[
            (name, training_seed)
        ]

    return baseline_runs[
        (name, training_seed)
    ]


def write_rows(
    output_path: Path,
    fieldnames: List[str],
    rows: List[dict],
) -> None:
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
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def run_pool_benchmark(
    pool_manifest_path: Path,
    baseline_manifest_path: Path,
    evaluation_seeds: List[int],
    games_per_seat: int,
    raw_output_path: Path,
    summary_output_path: Path,
    aggregate_output_path: Path,
) -> Tuple[List[dict], List[dict]]:
    if not evaluation_seeds:
        raise ValueError(
            "At least one evaluation seed is required."
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
        pool_runs,
        pool_training_seeds,
        pool_budget,
    ) = read_pool_training_manifest(
        pool_manifest_path
    )

    (
        baseline_runs,
        baseline_training_seeds,
        baseline_budget,
    ) = read_training_manifest(
        baseline_manifest_path
    )

    if (
        pool_training_seeds
        != baseline_training_seeds
    ):
        raise ValueError(
            "Pool and baseline runs must use "
            "the same training seeds."
        )

    if pool_budget != baseline_budget:
        raise ValueError(
            "Pool and baseline runs must use "
            "the same training budget."
        )

    reference_matchups = [
        (
            strategy,
            reference,
            "reference",
        )
        for strategy in POOL_STRATEGIES
        for reference in REFERENCE_AGENTS
    ]

    head_to_head_matchups = [
        (
            agent_a,
            agent_b,
            "head_to_head",
        )
        for agent_a, agent_b in (
            itertools.combinations(
                POOL_STRATEGIES,
                2,
            )
        )
    ]

    matchups = (
        reference_matchups
        + head_to_head_matchups
    )

    raw_rows: List[dict] = []
    summary_rows: List[dict] = []

    reference_scores: Dict[
        Tuple[str, int, str],
        float,
    ] = {}

    for (
        agent_a_name,
        agent_b_name,
        matchup_group,
    ) in matchups:
        training_seed_means = []

        for training_seed in (
            pool_training_seeds
        ):
            agent_a_spec = resolve_agent(
                name=agent_a_name,
                training_seed=training_seed,
                pool_runs=pool_runs,
                baseline_runs=baseline_runs,
            )
            agent_b_spec = resolve_agent(
                name=agent_b_name,
                training_seed=training_seed,
                pool_runs=pool_runs,
                baseline_runs=baseline_runs,
            )

            evaluation_scores = []

            for evaluation_seed in (
                evaluation_seeds
            ):
                (
                    seat_zero,
                    seat_one,
                ) = evaluate_both_seats(
                    agent_a_spec=agent_a_spec,
                    agent_b_spec=agent_b_spec,
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
                        "matchup_group": (
                            matchup_group
                        ),
                        "training_seed": (
                            training_seed
                        ),
                        "evaluation_seed": (
                            evaluation_seed
                        ),
                        "training_budget_seconds": (
                            pool_budget
                        ),
                        "agent_a": agent_a_name,
                        "agent_b": agent_b_name,
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

            if matchup_group == "reference":
                reference_scores[
                    (
                        agent_a_name,
                        training_seed,
                        agent_b_name,
                    )
                ] = training_seed_mean

            print(
                f"{agent_a_name} vs "
                f"{agent_b_name}, "
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
                "matchup_group": (
                    matchup_group
                ),
                "agent_a": agent_a_name,
                "agent_b": agent_b_name,
                "training_budget_seconds": (
                    pool_budget
                ),
                "training_seed_count": len(
                    pool_training_seeds
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
            f"{agent_a_name} vs "
            f"{agent_b_name}: "
            f"{mean_payoff:.4f}"
            f"{interval_text}"
        )

    aggregate_rows: List[dict] = []

    for strategy in POOL_STRATEGIES:
        seed_reference_means = []
        seed_worst_case_payoffs = []

        for training_seed in (
            pool_training_seeds
        ):
            scores = [
                reference_scores[
                    (
                        strategy,
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
            row["agent_b"]: float(
                row["mean_payoff"]
            )
            for row in summary_rows
            if (
                row["matchup_group"]
                == "reference"
                and row["agent_a"] == strategy
            )
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
                    "strategy": strategy,
                    "metric": metric,
                    "reference_agents": (
                        "|".join(
                            REFERENCE_AGENTS
                        )
                    ),
                    "training_seed_count": len(
                        pool_training_seeds
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
                f"{strategy} {metric}: "
                f"{mean_value:.4f}"
                f"{interval_text}"
            )

    write_rows(
        output_path=raw_output_path,
        fieldnames=[
            "matchup_group",
            "training_seed",
            "evaluation_seed",
            "training_budget_seconds",
            "agent_a",
            "agent_b",
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
            "matchup_group",
            "agent_a",
            "agent_b",
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
            "strategy",
            "metric",
            "reference_agents",
            "training_seed_count",
            "mean_payoff",
            "confidence_interval_95",
            "worst_opponent",
        ],
        rows=aggregate_rows,
    )

    print(
        f"Saved raw results to "
        f"{raw_output_path}"
    )
    print(
        f"Saved matchup summary to "
        f"{summary_output_path}"
    )
    print(
        f"Saved aggregate metrics to "
        f"{aggregate_output_path}"
    )

    return summary_rows, aggregate_rows