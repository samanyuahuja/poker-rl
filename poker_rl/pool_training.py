"""DQN training against a historical opponent pool."""

from __future__ import annotations

import csv
import statistics
import time
from copy import deepcopy
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List

import numpy as np
import rlcard
import torch

from rlcard.utils import reorganize, set_seed

from poker_rl.config import (
    environment_metadata,
    save_json,
)
from poker_rl.opponent_pool import (
    HistoricalOpponentPool,
)
from poker_rl.pool_config import (
    PoolTrainConfig,
    create_pool_run_directory,
    pool_config_as_dict,
)
from poker_rl.training import (
    NULL_WRITER,
    create_dqn_agent,
)

TRAINING_PROGRESS_FIELDS = [
    "strategy",
    "seed",
    "episode",
    "elapsed_seconds",
    "pool_size",
    "window_learner_payoff",
]

POOL_STATISTICS_FIELDS = [
    "snapshot_index",
    "episode",
    "elapsed_seconds",
    "checkpoint_path",
    "created_episode",
    "learner_wins",
    "learner_losses",
    "ties",
    "evaluation_count",
    "learner_loss_rate",
    "selection_count",
    "selection_probability",
]


def append_csv_rows(
    path: Path,
    fieldnames: List[str],
    rows: List[Dict[str, object]],
) -> None:
    if not rows:
        return

    write_header = not path.exists()

    with path.open("a", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        if write_header:
            writer.writeheader()

        writer.writerows(rows)


def create_frozen_agent(learner, env):
    """Copy only the learner policy into a greedy frozen agent."""

    frozen_agent = create_dqn_agent(env)

    frozen_agent.q_estimator = deepcopy(
        learner.q_estimator
    )
    frozen_agent.target_estimator = deepcopy(
        learner.target_estimator
    )

    frozen_agent.epsilons = np.zeros_like(
        frozen_agent.epsilons
    )
    frozen_agent.total_t = learner.total_t
    frozen_agent.train_t = learner.train_t

    return frozen_agent


def save_historical_checkpoint(
    learner,
    env,
    checkpoint_directory: Path,
    checkpoint_index: int,
):
    checkpoint_path = (
        checkpoint_directory
        / f"opponent_{checkpoint_index:04d}.pth"
    )

    frozen_agent = create_frozen_agent(
        learner,
        env,
    )

    torch.save(
        frozen_agent,
        checkpoint_path,
    )

    return checkpoint_path.resolve(), frozen_agent


def load_cached_opponent(
    checkpoint_path: Path,
    cache: Dict[Path, object],
):
    resolved_path = checkpoint_path.resolve()

    if resolved_path not in cache:
        cache[resolved_path] = torch.load(
            resolved_path,
            map_location="cpu",
            weights_only=False,
        )

    return cache[resolved_path]


def evaluate_against_pool(
    learner,
    pool: HistoricalOpponentPool,
    cache: Dict[Path, object],
    config: PoolTrainConfig,
    snapshot_index: int,
) -> None:
    """Update pool estimates without training either policy."""

    for opponent_index, record in enumerate(pool.records):
        opponent = load_cached_opponent(
            record.checkpoint_path,
            cache,
        )

        evaluation_environment = rlcard.make(
            "leduc-holdem",
            config={
                "seed": (
                    config.seed * 100000
                    + snapshot_index * 1000
                    + opponent_index
                ),
            },
        )

        for game_index in range(
            config.evaluation_games
        ):
            learner_seat = game_index % 2

            if learner_seat == 0:
                evaluation_environment.set_agents(
                    [learner, opponent]
                )
            else:
                evaluation_environment.set_agents(
                    [opponent, learner]
                )

            _, payoffs = evaluation_environment.run(
                is_training=False
            )

            record.record_payoff(
                float(payoffs[learner_seat])
            )


def write_pool_snapshot(
    path: Path,
    pool: HistoricalOpponentPool,
    run_directory: Path,
    snapshot_index: int,
    episode: int,
    elapsed_seconds: float,
) -> None:
    rows = []

    for pool_row in pool.rows():
        checkpoint_path = Path(
            str(pool_row["checkpoint_path"])
        )

        try:
            displayed_path = checkpoint_path.relative_to(
                run_directory
            )
        except ValueError:
            displayed_path = checkpoint_path

        row = dict(pool_row)
        row["checkpoint_path"] = str(displayed_path)
        row["snapshot_index"] = snapshot_index
        row["episode"] = episode
        row["elapsed_seconds"] = round(
            elapsed_seconds,
            6,
        )
        rows.append(row)

    append_csv_rows(
        path=path,
        fieldnames=POOL_STATISTICS_FIELDS,
        rows=rows,
    )


def training_budget_remains(
    config: PoolTrainConfig,
    start_time: float,
    completed_episodes: int,
) -> bool:
    if (
        time.perf_counter() - start_time
        >= config.budget_seconds
    ):
        return False

    if (
        config.max_episodes is not None
        and completed_episodes >= config.max_episodes
    ):
        return False

    return True


def train_with_opponent_pool(
    config: PoolTrainConfig,
) -> Path:
    """Train one DQN using the configured opponent sampler."""

    set_seed(config.seed)

    run_directory = create_pool_run_directory(
        config
    )
    checkpoint_directory = (
        run_directory / "historical_checkpoints"
    )
    checkpoint_directory.mkdir()

    save_json(
        run_directory / "config.json",
        pool_config_as_dict(config),
    )

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
        run_directory / "environment.json",
        environment,
    )

    training_environment = rlcard.make(
        "leduc-holdem",
        config={"seed": config.seed},
    )
    learner = create_dqn_agent(
        training_environment
    )

    pool = HistoricalOpponentPool(
        strategy=config.strategy,
        seed=config.seed,
        max_size=config.max_pool_size,
        beta=config.beta,
        epsilon=config.epsilon,
        temperature=config.temperature,
    )

    opponent_cache: Dict[Path, object] = {}

    progress_path = (
        run_directory / "training_progress.csv"
    )
    pool_statistics_path = (
        run_directory / "pool_statistics.csv"
    )

    initial_path, initial_opponent = (
        save_historical_checkpoint(
            learner=learner,
            env=training_environment,
            checkpoint_directory=checkpoint_directory,
            checkpoint_index=0,
        )
    )

    pool.add_checkpoint(
        initial_path,
        created_episode=0,
    )
    opponent_cache[initial_path] = initial_opponent

    write_pool_snapshot(
        path=pool_statistics_path,
        pool=pool,
        run_directory=run_directory,
        snapshot_index=0,
        episode=0,
        elapsed_seconds=0,
    )

    completed_episodes = 0
    checkpoint_index = 0
    next_checkpoint_time = (
        config.checkpoint_seconds
    )
    payoff_window: List[float] = []

    start_time = time.perf_counter()

    try:
        while training_budget_remains(
            config,
            start_time,
            completed_episodes,
        ):
            opponent_record = pool.sample()
            opponent = load_cached_opponent(
                opponent_record.checkpoint_path,
                opponent_cache,
            )

            learner_seat = completed_episodes % 2

            if learner_seat == 0:
                training_environment.set_agents(
                    [learner, opponent]
                )
            else:
                training_environment.set_agents(
                    [opponent, learner]
                )

            trajectories, payoffs = (
                training_environment.run(
                    is_training=True
                )
            )
            trajectories = reorganize(
                trajectories,
                payoffs,
            )

            with redirect_stdout(NULL_WRITER):
                for transition in trajectories[
                    learner_seat
                ]:
                    learner.feed(transition)

            learner_payoff = float(
                payoffs[learner_seat]
            )
            payoff_window.append(learner_payoff)
            completed_episodes += 1

            elapsed = (
                time.perf_counter() - start_time
            )

            if (
                completed_episodes
                % config.log_every
                == 0
            ):
                append_csv_rows(
                    path=progress_path,
                    fieldnames=(
                        TRAINING_PROGRESS_FIELDS
                    ),
                    rows=[
                        {
                            "strategy": config.strategy,
                            "seed": config.seed,
                            "episode": (
                                completed_episodes
                            ),
                            "elapsed_seconds": round(
                                elapsed,
                                6,
                            ),
                            "pool_size": len(pool),
                            "window_learner_payoff": (
                                statistics.mean(
                                    payoff_window
                                )
                            ),
                        }
                    ],
                )

                print(
                    f"{config.strategy}: "
                    f"{completed_episodes} episodes, "
                    f"pool size {len(pool)}, "
                    f"{elapsed:.1f} seconds"
                )
                payoff_window.clear()

            if (
                elapsed >= next_checkpoint_time
                and elapsed < config.budget_seconds
            ):
                checkpoint_index += 1

                evaluate_against_pool(
                    learner=learner,
                    pool=pool,
                    cache=opponent_cache,
                    config=config,
                    snapshot_index=checkpoint_index,
                )

                elapsed = (
                    time.perf_counter() - start_time
                )

                write_pool_snapshot(
                    path=pool_statistics_path,
                    pool=pool,
                    run_directory=run_directory,
                    snapshot_index=checkpoint_index,
                    episode=completed_episodes,
                    elapsed_seconds=elapsed,
                )

                checkpoint_path, frozen_agent = (
                    save_historical_checkpoint(
                        learner=learner,
                        env=training_environment,
                        checkpoint_directory=(
                            checkpoint_directory
                        ),
                        checkpoint_index=(
                            checkpoint_index
                        ),
                    )
                )

                removed_record = pool.add_checkpoint(
                    checkpoint_path,
                    created_episode=(
                        completed_episodes
                    ),
                )
                opponent_cache[
                    checkpoint_path
                ] = frozen_agent

                if removed_record is not None:
                    opponent_cache.pop(
                        removed_record.checkpoint_path,
                        None,
                    )

                print(
                    f"Added checkpoint "
                    f"{checkpoint_index} at episode "
                    f"{completed_episodes}"
                )

                while (
                    next_checkpoint_time <= elapsed
                ):
                    next_checkpoint_time += (
                        config.checkpoint_seconds
                    )

        training_elapsed = (
            time.perf_counter() - start_time
        )

        if payoff_window:
            append_csv_rows(
                path=progress_path,
                fieldnames=(
                    TRAINING_PROGRESS_FIELDS
                ),
                rows=[
                    {
                        "strategy": config.strategy,
                        "seed": config.seed,
                        "episode": completed_episodes,
                        "elapsed_seconds": round(
                            training_elapsed,
                            6,
                        ),
                        "pool_size": len(pool),
                        "window_learner_payoff": (
                            statistics.mean(
                                payoff_window
                            )
                        ),
                    }
                ],
            )

        final_evaluation_start = (
            time.perf_counter()
        )

        evaluate_against_pool(
            learner=learner,
            pool=pool,
            cache=opponent_cache,
            config=config,
            snapshot_index=checkpoint_index + 1,
        )

        write_pool_snapshot(
            path=pool_statistics_path,
            pool=pool,
            run_directory=run_directory,
            snapshot_index=checkpoint_index + 1,
            episode=completed_episodes,
            elapsed_seconds=training_elapsed,
        )

        final_agent = create_frozen_agent(
            learner,
            training_environment,
        )

        torch.save(
            final_agent,
            run_directory / "player_0.pth",
        )
        torch.save(
            final_agent,
            run_directory / "player_1.pth",
        )

        final_evaluation_seconds = (
            time.perf_counter()
            - final_evaluation_start
        )

        save_json(
            run_directory / "summary.json",
            {
                "status": "complete",
                "algorithm": "dqn",
                "strategy": config.strategy,
                "unit_name": "episode",
                "completed_units": (
                    completed_episodes
                ),
                "training_elapsed_seconds": (
                    training_elapsed
                ),
                "post_training_evaluation_seconds": (
                    final_evaluation_seconds
                ),
                "historical_checkpoint_count": (
                    checkpoint_index + 1
                ),
                "final_pool_size": len(pool),
                "checkpoints": [
                    "player_0.pth",
                    "player_1.pth",
                    "historical_checkpoints/",
                ],
            },
        )

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

    print(
        f"Completed pool-training run: "
        f"{run_directory}"
    )

    return run_directory