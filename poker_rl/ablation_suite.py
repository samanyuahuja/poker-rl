"""Matched-budget ablation training for uncertainty-aware self-play."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from poker_rl.config import (
    ARTIFACTS_DIR,
    PROJECT_ROOT,
)
from poker_rl.pool_config import PoolTrainConfig
from poker_rl.pool_suite import (
    validate_existing_pool_run,
)
from poker_rl.pool_training import (
    train_with_opponent_pool,
)
from poker_rl.suite import (
    budget_slug,
    read_json,
)


@dataclass(frozen=True)
class AblationSpec:
    label: str
    family: str
    beta: float
    max_pool_size: int
    checkpoint_seconds: float


ABLATION_SPECS: Tuple[AblationSpec, ...] = (
    AblationSpec(
        label="beta-0",
        family="uncertainty_strength",
        beta=0.0,
        max_pool_size=10,
        checkpoint_seconds=30.0,
    ),
    AblationSpec(
        label="beta-2",
        family="uncertainty_strength",
        beta=2.0,
        max_pool_size=10,
        checkpoint_seconds=30.0,
    ),
    AblationSpec(
        label="pool-5",
        family="pool_size",
        beta=1.0,
        max_pool_size=5,
        checkpoint_seconds=10.0,
    ),
    AblationSpec(
        label="pool-10",
        family="pool_size",
        beta=1.0,
        max_pool_size=10,
        checkpoint_seconds=10.0,
    ),
    AblationSpec(
        label="pool-20",
        family="pool_size",
        beta=1.0,
        max_pool_size=20,
        checkpoint_seconds=10.0,
    ),
)

ABLATION_BY_LABEL = {
    spec.label: spec
    for spec in ABLATION_SPECS
}


def ablation_run_name(
    label: str,
    seed: int,
    budget_seconds: float,
) -> str:
    return (
        f"ablation-{label}-seed{seed}-"
        f"budget{budget_slug(budget_seconds)}s"
    )


def validate_labels(
    labels: List[str],
) -> List[AblationSpec]:
    if not labels:
        raise ValueError(
            "At least one ablation label is required."
        )

    if len(labels) != len(set(labels)):
        raise ValueError(
            "Ablation labels must not contain "
            "duplicates."
        )

    unknown_labels = [
        label
        for label in labels
        if label not in ABLATION_BY_LABEL
    ]

    if unknown_labels:
        raise ValueError(
            "Unknown ablation labels: "
            + ", ".join(
                sorted(unknown_labels)
            )
        )

    return [
        ABLATION_BY_LABEL[label]
        for label in labels
    ]


def run_ablation_suite(
    labels: List[str],
    seeds: List[int],
    budget_seconds: float,
    evaluation_games: int,
    epsilon: float,
    temperature: float,
    log_every: int,
    output_path: Path,
) -> List[Dict[str, Any]]:
    specs = validate_labels(labels)

    if not seeds:
        raise ValueError(
            "At least one training seed is required."
        )

    if len(seeds) != len(set(seeds)):
        raise ValueError(
            "Training seeds must not contain "
            "duplicates."
        )

    rows: List[Dict[str, Any]] = []

    for seed in seeds:
        for spec in specs:
            run_name = ablation_run_name(
                label=spec.label,
                seed=seed,
                budget_seconds=budget_seconds,
            )

            config = PoolTrainConfig(
                strategy="uncertainty",
                seed=seed,
                budget_seconds=budget_seconds,
                checkpoint_seconds=(
                    spec.checkpoint_seconds
                ),
                evaluation_games=evaluation_games,
                max_pool_size=(
                    spec.max_pool_size
                ),
                beta=spec.beta,
                epsilon=epsilon,
                temperature=temperature,
                log_every=log_every,
                run_name=run_name,
            )

            run_directory = (
                ARTIFACTS_DIR / run_name
            )

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
                    train_with_opponent_pool(
                        config
                    )
                )
                action = "trained"

            summary = read_json(
                run_directory / "summary.json"
            )

            rows.append(
                {
                    "ablation_label": (
                        spec.label
                    ),
                    "ablation_family": (
                        spec.family
                    ),
                    "strategy": "uncertainty",
                    "training_seed": seed,
                    "budget_seconds": (
                        budget_seconds
                    ),
                    "checkpoint_seconds": (
                        spec.checkpoint_seconds
                    ),
                    "evaluation_games": (
                        evaluation_games
                    ),
                    "max_pool_size": (
                        spec.max_pool_size
                    ),
                    "beta": spec.beta,
                    "epsilon": epsilon,
                    "temperature": temperature,
                    "completed_episodes": (
                        summary[
                            "completed_units"
                        ]
                    ),
                    "training_elapsed_seconds": (
                        summary[
                            "training_elapsed_seconds"
                        ]
                    ),
                    "historical_checkpoint_count": (
                        summary[
                            "historical_checkpoint_count"
                        ]
                    ),
                    "final_pool_size": (
                        summary[
                            "final_pool_size"
                        ]
                    ),
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
        "ablation_label",
        "ablation_family",
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
        f"Saved ablation training manifest "
        f"to {output_path}"
    )

    return rows