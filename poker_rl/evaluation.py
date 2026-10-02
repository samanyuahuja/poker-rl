"""Reproducible evaluation of saved poker agents."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import rlcard
import torch

from rlcard.agents import CFRAgent, RandomAgent
from rlcard.utils import set_seed, tournament

AgentSpec = Union[str, Path]

T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open() as input_file:
        return json.load(input_file)


def agent_label(spec: AgentSpec) -> str:
    if str(spec) == "random":
        return "random"

    return Path(spec).resolve().name


def load_agent(
    spec: AgentSpec,
    seat: int,
    env,
):
    if str(spec) == "random":
        return RandomAgent(num_actions=env.num_actions)

    run_directory = Path(spec).resolve()

    config_path = run_directory / "config.json"
    summary_path = run_directory / "summary.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing run configuration: {config_path}"
        )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing run summary: {summary_path}"
        )

    config = read_json(config_path)
    summary = read_json(summary_path)

    if summary.get("status") != "complete":
        raise ValueError(
            f"Run is not complete: {run_directory}"
        )

    algorithm = config["algorithm"]

    if algorithm in {"dqn", "nfsp"}:
        checkpoint_path = (
            run_directory / f"player_{seat}.pth"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {checkpoint_path}"
            )

        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

    if algorithm == "cfr":
        checkpoint_directory = run_directory / "cfr"

        agent = CFRAgent(
            env,
            model_path=str(checkpoint_directory),
        )
        agent.load()
        return agent

    raise ValueError(
        f"Unsupported algorithm in {config_path}: "
        f"{algorithm}"
    )


def confidence_interval_95(
    values: List[float],
) -> Optional[float]:
    if len(values) < 2:
        return None

    degrees_of_freedom = len(values) - 1
    critical_value = T_CRITICAL_95.get(
        degrees_of_freedom,
        1.96,
    )

    standard_error = (
        statistics.stdev(values)
        / math.sqrt(len(values))
    )

    return critical_value * standard_error


def evaluate_matchup(
    agent_a_spec: AgentSpec,
    agent_b_spec: AgentSpec,
    seeds: List[int],
    games_per_seat: int,
    output_path: Path,
) -> Dict[str, Any]:
    if not seeds:
        raise ValueError(
            "At least one evaluation seed is required."
        )

    if games_per_seat <= 0:
        raise ValueError(
            "games_per_seat must be greater than zero."
        )

    agent_a_name = agent_label(agent_a_spec)
    agent_b_name = agent_label(agent_b_spec)

    rows = []
    combined_scores = []

    for seed in seeds:
        set_seed(seed)

        first_env = rlcard.make(
            "leduc-holdem",
            config={
                "seed": seed,
                "allow_step_back": True,
            },
        )

        first_agent_a = load_agent(
            agent_a_spec,
            seat=0,
            env=first_env,
        )
        first_agent_b = load_agent(
            agent_b_spec,
            seat=1,
            env=first_env,
        )

        first_env.set_agents(
            [first_agent_a, first_agent_b]
        )
        first_payoff = float(
            tournament(
                first_env,
                games_per_seat,
            )[0]
        )

        set_seed(seed)

        second_env = rlcard.make(
            "leduc-holdem",
            config={
                "seed": seed,
                "allow_step_back": True,
            },
        )

        second_agent_b = load_agent(
            agent_b_spec,
            seat=0,
            env=second_env,
        )
        second_agent_a = load_agent(
            agent_a_spec,
            seat=1,
            env=second_env,
        )

        second_env.set_agents(
            [second_agent_b, second_agent_a]
        )
        second_payoff = float(
            tournament(
                second_env,
                games_per_seat,
            )[1]
        )

        combined_payoff = (
            first_payoff + second_payoff
        ) / 2

        rows.extend(
            [
                {
                    "agent_a": agent_a_name,
                    "agent_b": agent_b_name,
                    "evaluation_seed": seed,
                    "games": games_per_seat,
                    "agent_a_seat": 0,
                    "agent_a_average_payoff": first_payoff,
                },
                {
                    "agent_a": agent_a_name,
                    "agent_b": agent_b_name,
                    "evaluation_seed": seed,
                    "games": games_per_seat,
                    "agent_a_seat": 1,
                    "agent_a_average_payoff": second_payoff,
                },
            ]
        )

        combined_scores.append(combined_payoff)

        print(
            f"{agent_a_name} vs {agent_b_name}, "
            f"seed {seed}: "
            f"seat 0={first_payoff:.4f}, "
            f"seat 1={second_payoff:.4f}, "
            f"combined={combined_payoff:.4f}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "agent_a",
                "agent_b",
                "evaluation_seed",
                "games",
                "agent_a_seat",
                "agent_a_average_payoff",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    mean_payoff = statistics.mean(combined_scores)
    interval = confidence_interval_95(
        combined_scores
    )

    summary = {
        "agent_a": agent_a_name,
        "agent_b": agent_b_name,
        "evaluation_seeds": seeds,
        "games_per_seat": games_per_seat,
        "mean_payoff": mean_payoff,
        "confidence_interval_95": interval,
        "output_path": str(output_path),
    }

    print(
        f"\n{agent_a_name} vs {agent_b_name}: "
        f"{mean_payoff:.4f}"
        + (
            ""
            if interval is None
            else f" ± {interval:.4f}"
        )
    )
    print(f"Saved evaluation to {output_path}")

    return summary