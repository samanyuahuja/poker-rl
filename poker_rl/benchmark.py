"""Matched-budget evaluation for all baseline poker agents."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Union

import rlcard

from rlcard.utils import set_seed, tournament

from poker_rl.config import PROJECT_ROOT
from poker_rl.evaluation import (
    confidence_interval_95,
    load_agent,
)

AgentSpec = Union[str, Path]

TRAINED_ALGORITHMS = ("dqn", "nfsp", "cfr")

MATCHUPS = (
    ("dqn", "random"),
    ("nfsp", "random"),
    ("cfr", "random"),
    ("dqn", "nfsp"),
    ("dqn", "cfr"),
    ("nfsp", "cfr"),
)


def read_training_manifest(
    manifest_path: Path,
) -> Tuple[Dict[Tuple[str, int], Path], List[int], float]:
    """Load and validate matched-budget training runs."""

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Training manifest does not exist: {manifest_path}"
        )

    runs: Dict[Tuple[str, int], Path] = {}
    budgets = set()

    with manifest_path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)

        required_fields = {
            "algorithm",
            "training_seed",
            "budget_seconds",
            "run_directory",
        }

        if reader.fieldnames is None:
            raise ValueError("Training manifest has no header.")

        missing_fields = required_fields - set(reader.fieldnames)

        if missing_fields:
            raise ValueError(
                "Training manifest is missing fields: "
                + ", ".join(sorted(missing_fields))
            )

        for row in reader:
            algorithm = row["algorithm"]
            training_seed = int(row["training_seed"])
            budget_seconds = float(row["budget_seconds"])

            if algorithm not in TRAINED_ALGORITHMS:
                continue

            run_directory = Path(row["run_directory"])

            if not run_directory.is_absolute():
                run_directory = PROJECT_ROOT / run_directory

            key = (algorithm, training_seed)

            if key in runs:
                raise ValueError(
                    "Duplicate training run for "
                    f"{algorithm}, seed {training_seed}."
                )

            runs[key] = run_directory.resolve()
            budgets.add(budget_seconds)

    if len(budgets) != 1:
        raise ValueError(
            "All training runs must use one common time budget."
        )

    algorithm_seed_sets = {
        algorithm: {
            seed
            for stored_algorithm, seed in runs
            if stored_algorithm == algorithm
        }
        for algorithm in TRAINED_ALGORITHMS
    }

    reference_seeds = algorithm_seed_sets["dqn"]

    if not reference_seeds:
        raise ValueError("No DQN training runs were found.")

    for algorithm, seeds in algorithm_seed_sets.items():
        if seeds != reference_seeds:
            raise ValueError(
                "Every algorithm must have the same training seeds. "
                f"{algorithm} has {sorted(seeds)}, while DQN has "
                f"{sorted(reference_seeds)}."
            )

    return (
        runs,
        sorted(reference_seeds),
        budgets.pop(),
    )


def evaluate_both_seats(
    agent_a_spec: AgentSpec,
    agent_b_spec: AgentSpec,
    evaluation_seed: int,
    games_per_seat: int,
) -> Tuple[float, float]:
    """Evaluate agent A once from each player position."""

    set_seed(evaluation_seed)

    first_environment = rlcard.make(
        "leduc-holdem",
        config={
            "seed": evaluation_seed,
            "allow_step_back": True,
        },
    )

    first_agent_a = load_agent(
        agent_a_spec,
        seat=0,
        env=first_environment,
    )
    first_agent_b = load_agent(
        agent_b_spec,
        seat=1,
        env=first_environment,
    )

    first_environment.set_agents(
        [first_agent_a, first_agent_b]
    )

    seat_zero_payoff = float(
        tournament(
            first_environment,
            games_per_seat,
        )[0]
    )

    set_seed(evaluation_seed)

    second_environment = rlcard.make(
        "leduc-holdem",
        config={
            "seed": evaluation_seed,
            "allow_step_back": True,
        },
    )

    second_agent_b = load_agent(
        agent_b_spec,
        seat=0,
        env=second_environment,
    )
    second_agent_a = load_agent(
        agent_a_spec,
        seat=1,
        env=second_environment,
    )

    second_environment.set_agents(
        [second_agent_b, second_agent_a]
    )

    seat_one_payoff = float(
        tournament(
            second_environment,
            games_per_seat,
        )[1]
    )

    return seat_zero_payoff, seat_one_payoff


def run_baseline_benchmark(
    manifest_path: Path,
    evaluation_seeds: List[int],
    games_per_seat: int,
    raw_output_path: Path,
    summary_output_path: Path,
) -> List[dict]:
    """Evaluate every baseline matchup across training seeds."""

    if not evaluation_seeds:
        raise ValueError(
            "At least one evaluation seed is required."
        )

    if games_per_seat <= 0:
        raise ValueError(
            "games_per_seat must be greater than zero."
        )

    runs, training_seeds, budget_seconds = (
        read_training_manifest(manifest_path)
    )

    raw_rows = []
    summary_rows = []

    for agent_a_name, agent_b_name in MATCHUPS:
        training_seed_means = []

        for training_seed in training_seeds:
            agent_a_spec: AgentSpec = runs[
                (agent_a_name, training_seed)
            ]

            if agent_b_name == "random":
                agent_b_spec: AgentSpec = "random"
            else:
                agent_b_spec = runs[
                    (agent_b_name, training_seed)
                ]

            evaluation_scores = []

            for evaluation_seed in evaluation_seeds:
                seat_zero, seat_one = evaluate_both_seats(
                    agent_a_spec=agent_a_spec,
                    agent_b_spec=agent_b_spec,
                    evaluation_seed=evaluation_seed,
                    games_per_seat=games_per_seat,
                )

                combined_payoff = (
                    seat_zero + seat_one
                ) / 2

                evaluation_scores.append(combined_payoff)

                raw_rows.append(
                    {
                        "training_seed": training_seed,
                        "evaluation_seed": evaluation_seed,
                        "training_budget_seconds": (
                            budget_seconds
                        ),
                        "agent_a": agent_a_name,
                        "agent_b": agent_b_name,
                        "games_per_seat": games_per_seat,
                        "seat_0_payoff": seat_zero,
                        "seat_1_payoff": seat_one,
                        "combined_payoff": combined_payoff,
                    }
                )

            training_seed_mean = statistics.mean(
                evaluation_scores
            )
            training_seed_means.append(training_seed_mean)

            print(
                f"{agent_a_name} vs {agent_b_name}, "
                f"training seed {training_seed}: "
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
                "agent_a": agent_a_name,
                "agent_b": agent_b_name,
                "training_budget_seconds": budget_seconds,
                "training_seed_count": len(training_seeds),
                "evaluation_seed_count": len(
                    evaluation_seeds
                ),
                "games_per_seat": games_per_seat,
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
            f"{agent_a_name} vs {agent_b_name}: "
            f"{mean_payoff:.4f}{interval_text}"
        )

    raw_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with raw_output_path.open(
        "w",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
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
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    with summary_output_path.open(
        "w",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "agent_a",
                "agent_b",
                "training_budget_seconds",
                "training_seed_count",
                "evaluation_seed_count",
                "games_per_seat",
                "mean_payoff",
                "confidence_interval_95",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved raw results to {raw_output_path}")
    print(f"Saved summary to {summary_output_path}")

    return summary_rows