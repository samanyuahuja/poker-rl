"""Run matched-budget opponent-pool training across strategies and seeds."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from poker_rl.config import (
    ARTIFACTS_DIR,
    PROJECT_ROOT,
)
from poker_rl.pool_config import (
    PoolTrainConfig,
    pool_config_as_dict,
)
from poker_rl.pool_training import (
    train_with_opponent_pool,
)
from poker_rl.suite import (
    budget_slug,
    read_json,
)


def pool_run_name(
    strategy: str,
    seed: int,
    budget_seconds: float,
) -> str:
    return (
        f"pool-{strategy}-seed{seed}-"
        f"budget{budget_slug(budget_seconds)}s"
    )


def validate_existing_pool_run(
    run_directory: Path,
    config: PoolTrainConfig,
) -> None:
    config_path = run_directory / "config.json"
    summary_path = run_directory / "summary.json"

    required_paths = [
        config_path,
        summary_path,
        run_directory / "player_0.pth",
        run_directory / "player_1.pth",
        run_directory / "training_progress.csv",
        run_directory / "pool_statistics.csv",
    ]

    missing_paths = [
        path.name
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise ValueError(
            f"Incomplete existing run "
            f"{run_directory}. Missing: "
            f"{', '.join(missing_paths)}."
        )

    existing_config = read_json(config_path)
    expected_config = pool_config_as_dict(config)

    if existing_config != expected_config:
        raise ValueError(
            f"Existing run configuration mismatch in "
            f"{run_directory}."
        )

    summary = read_json(summary_path)

    if summary.get("status") != "complete":
        raise ValueError(
            f"Existing run did not complete: "
            f"{run_directory}"
        )


def run_pool_training_suite(
    strategies: List[str],
    seeds: List[int],
    budget_seconds: float,
    checkpoint_seconds: float,
    evaluation_games: int,
    max_pool_size: int,
    beta: float,
    epsilon: float,
    temperature: float,
    log_every: int,
    output_path: Path,
) -> List[Dict[str, Any]]:
    if not strategies:
        raise ValueError(
            "At least one strategy is required."
        )

    if len(strategies) != len(set(strategies)):
        raise ValueError(
            "Strategies must not contain duplicates."
        )

    if not seeds:
        raise ValueError(
            "At least one training seed is required."
        )

    if len(seeds) != len(set(seeds)):
        raise ValueError(
            "Training seeds must not contain duplicates."
        )

    rows: List[Dict[str, Any]] = []

    for seed in seeds:
        for strategy in strategies:
            run_name = pool_run_name(
                strategy=strategy,
                seed=seed,
                budget_seconds=budget_seconds,
            )

            config = PoolTrainConfig(
                strategy=strategy,
                seed=seed,
                budget_seconds=budget_seconds,
                checkpoint_seconds=checkpoint_seconds,
                evaluation_games=evaluation_games,
                max_pool_size=max_pool_size,
                beta=beta,
                epsilon=epsilon,
                temperature=temperature,
                log_every=log_every,
                run_name=run_name,
            )

            run_directory = ARTIFACTS_DIR / run_name

            if run_directory.exists():
                validate_existing_pool_run(
                    run_directory=run_directory,
                    config=config,
                )
                action = "reused"

                print(
                    f"Reusing completed run: "
                    f"{run_directory}"
                )
            else:
                run_directory = (
                    train_with_opponent_pool(config)
                )
                action = "trained"

            summary = read_json(
                run_directory / "summary.json"
            )

            rows.append(
                {
                    "strategy": strategy,
                    "training_seed": seed,
                    "budget_seconds": budget_seconds,
                    "checkpoint_seconds": (
                        checkpoint_seconds
                    ),
                    "evaluation_games": (
                        evaluation_games
                    ),
                    "max_pool_size": max_pool_size,
                    "beta": beta,
                    "epsilon": epsilon,
                    "temperature": temperature,
                    "completed_episodes": summary[
                        "completed_units"
                    ],
                    "training_elapsed_seconds": summary[
                        "training_elapsed_seconds"
                    ],
                    "post_training_evaluation_seconds": (
                        summary[
                            "post_training_evaluation_seconds"
                        ]
                    ),
                    "historical_checkpoint_count": (
                        summary[
                            "historical_checkpoint_count"
                        ]
                    ),
                    "final_pool_size": summary[
                        "final_pool_size"
                    ],
                    "run_directory": str(
                        run_directory.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "action": action,
                }
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "strategy",
        "training_seed",
        "budget_seconds",
        "checkpoint_seconds",
        "evaluation_games",
        "max_pool_size",
        "beta",
        "epsilon",
        "temperature",
        "completed_episodes",
        "training_elapsed_seconds",
        "post_training_evaluation_seconds",
        "historical_checkpoint_count",
        "final_pool_size",
        "run_directory",
        "action",
    ]

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

    print(
        f"Saved opponent-pool training manifest "
        f"to {output_path}"
    )

    return rows