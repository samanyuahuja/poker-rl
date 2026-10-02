"""Time-budgeted training for the baseline poker agents."""

from __future__ import annotations

import csv
import io
import statistics
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional

import rlcard
import torch

from rlcard.agents import CFRAgent, DQNAgent, NFSPAgent
from rlcard.utils import reorganize, set_seed

from poker_rl.config import (
    TrainConfig,
    config_as_dict,
    create_run_directory,
    environment_metadata,
    save_json,
)


class NullWriter(io.TextIOBase):
    """Discard noisy progress messages printed internally by RLCard."""

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        return None


NULL_WRITER = NullWriter()

PROGRESS_FIELDS = [
    "algorithm",
    "seed",
    "unit_name",
    "unit",
    "elapsed_seconds",
    "window_payoff_player_0",
]


def budget_remaining(
    config: TrainConfig,
    start_time: float,
    completed_units: int,
) -> bool:
    elapsed = time.perf_counter() - start_time

    if elapsed >= config.budget_seconds:
        return False

    if (
        config.max_units is not None
        and completed_units >= config.max_units
    ):
        return False

    return True


def record_progress(
    path: Path,
    config: TrainConfig,
    unit_name: str,
    unit: int,
    elapsed_seconds: float,
    window_payoff: Optional[float],
) -> None:
    write_header = not path.exists()

    with path.open("a", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=PROGRESS_FIELDS,
        )

        if write_header:
            writer.writeheader()

        writer.writerow(
            {
                "algorithm": config.algorithm,
                "seed": config.seed,
                "unit_name": unit_name,
                "unit": unit,
                "elapsed_seconds": round(elapsed_seconds, 6),
                "window_payoff_player_0": (
                    ""
                    if window_payoff is None
                    else round(window_payoff, 6)
                ),
            }
        )


def create_dqn_agent(env) -> DQNAgent:
    return DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[64, 64],
        device=torch.device("cpu"),
    )


def create_nfsp_agent(env) -> NFSPAgent:
    return NFSPAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        hidden_layers_sizes=[64, 64],
        q_mlp_layers=[64, 64],
        device=torch.device("cpu"),
    )


def train_dqn(
    config: TrainConfig,
    run_directory: Path,
) -> Dict[str, Any]:
    env = rlcard.make(
        "leduc-holdem",
        config={"seed": config.seed},
    )

    agents = [
        create_dqn_agent(env),
        create_dqn_agent(env),
    ]
    env.set_agents(agents)

    progress_path = run_directory / "training_progress.csv"
    payoff_window: List[float] = []
    completed_episodes = 0
    start_time = time.perf_counter()

    while budget_remaining(
        config,
        start_time,
        completed_episodes,
    ):
        trajectories, payoffs = env.run(is_training=True)
        trajectories = reorganize(trajectories, payoffs)

        with redirect_stdout(NULL_WRITER):
            for player_id, agent in enumerate(agents):
                for transition in trajectories[player_id]:
                    agent.feed(transition)

        completed_episodes += 1
        payoff_window.append(float(payoffs[0]))

        if completed_episodes % config.log_every == 0:
            elapsed = time.perf_counter() - start_time
            mean_payoff = statistics.mean(payoff_window)

            record_progress(
                progress_path,
                config,
                "episode",
                completed_episodes,
                elapsed,
                mean_payoff,
            )

            print(
                f"DQN: {completed_episodes} episodes "
                f"in {elapsed:.1f} seconds"
            )
            payoff_window.clear()

    training_elapsed = time.perf_counter() - start_time

    if payoff_window:
        record_progress(
            progress_path,
            config,
            "episode",
            completed_episodes,
            training_elapsed,
            statistics.mean(payoff_window),
        )

    checkpoint_paths = [
        run_directory / "player_0.pth",
        run_directory / "player_1.pth",
    ]

    torch.save(agents[0], checkpoint_paths[0])
    torch.save(agents[1], checkpoint_paths[1])

    return {
        "unit_name": "episode",
        "completed_units": completed_episodes,
        "training_elapsed_seconds": training_elapsed,
        "checkpoints": [
            path.name
            for path in checkpoint_paths
        ],
    }


def train_nfsp(
    config: TrainConfig,
    run_directory: Path,
) -> Dict[str, Any]:
    env = rlcard.make(
        "leduc-holdem",
        config={"seed": config.seed},
    )

    agents = [
        create_nfsp_agent(env),
        create_nfsp_agent(env),
    ]
    env.set_agents(agents)

    progress_path = run_directory / "training_progress.csv"
    payoff_window: List[float] = []
    completed_episodes = 0
    start_time = time.perf_counter()

    while budget_remaining(
        config,
        start_time,
        completed_episodes,
    ):
        for agent in agents:
            agent.sample_episode_policy()

        trajectories, payoffs = env.run(is_training=True)
        trajectories = reorganize(trajectories, payoffs)

        with redirect_stdout(NULL_WRITER):
            for player_id, agent in enumerate(agents):
                for transition in trajectories[player_id]:
                    agent.feed(transition)

        completed_episodes += 1
        payoff_window.append(float(payoffs[0]))

        if completed_episodes % config.log_every == 0:
            elapsed = time.perf_counter() - start_time
            mean_payoff = statistics.mean(payoff_window)

            record_progress(
                progress_path,
                config,
                "episode",
                completed_episodes,
                elapsed,
                mean_payoff,
            )

            print(
                f"NFSP: {completed_episodes} episodes "
                f"in {elapsed:.1f} seconds"
            )
            payoff_window.clear()

    training_elapsed = time.perf_counter() - start_time

    if payoff_window:
        record_progress(
            progress_path,
            config,
            "episode",
            completed_episodes,
            training_elapsed,
            statistics.mean(payoff_window),
        )

    checkpoint_paths = [
        run_directory / "player_0.pth",
        run_directory / "player_1.pth",
    ]

    torch.save(agents[0], checkpoint_paths[0])
    torch.save(agents[1], checkpoint_paths[1])

    return {
        "unit_name": "episode",
        "completed_units": completed_episodes,
        "training_elapsed_seconds": training_elapsed,
        "checkpoints": [
            path.name
            for path in checkpoint_paths
        ],
    }


def train_cfr(
    config: TrainConfig,
    run_directory: Path,
) -> Dict[str, Any]:
    env = rlcard.make(
        "leduc-holdem",
        config={
            "seed": config.seed,
            "allow_step_back": True,
        },
    )

    checkpoint_directory = run_directory / "cfr"
    agent = CFRAgent(
        env,
        model_path=str(checkpoint_directory),
    )

    progress_path = run_directory / "training_progress.csv"
    completed_iterations = 0
    start_time = time.perf_counter()

    while budget_remaining(
        config,
        start_time,
        completed_iterations,
    ):
        agent.train()
        completed_iterations += 1

        if completed_iterations % config.log_every == 0:
            elapsed = time.perf_counter() - start_time

            record_progress(
                progress_path,
                config,
                "iteration",
                completed_iterations,
                elapsed,
                None,
            )

            print(
                f"CFR: {completed_iterations} iterations "
                f"in {elapsed:.1f} seconds"
            )

    training_elapsed = time.perf_counter() - start_time

    if completed_iterations % config.log_every != 0:
        record_progress(
            progress_path,
            config,
            "iteration",
            completed_iterations,
            training_elapsed,
            None,
        )

    agent.save()

    return {
        "unit_name": "iteration",
        "completed_units": completed_iterations,
        "training_elapsed_seconds": training_elapsed,
        "checkpoints": ["cfr/"],
    }


TRAINERS = {
    "dqn": train_dqn,
    "nfsp": train_nfsp,
    "cfr": train_cfr,
}


def train(config: TrainConfig) -> Path:
    """Train one algorithm and save a reproducible run artifact."""

    set_seed(config.seed)
    run_directory = create_run_directory(config)

    environment = environment_metadata()
    environment.update(
        {
            "torch_version": torch.__version__,
            "rlcard_version": getattr(
                rlcard,
                "__version__",
                "unknown",
            ),
            "device": "cpu",
        }
    )

    save_json(
        run_directory / "config.json",
        config_as_dict(config),
    )
    save_json(
        run_directory / "environment.json",
        environment,
    )

    try:
        summary = TRAINERS[config.algorithm](
            config,
            run_directory,
        )
        summary["status"] = "complete"

    except Exception as error:
        save_json(
            run_directory / "summary.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise

    save_json(
        run_directory / "summary.json",
        summary,
    )

    print(f"Completed run: {run_directory}")
    return run_directory