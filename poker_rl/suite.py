"""Run matched-budget training across algorithms and seeds."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from poker_rl.config import (
    ARTIFACTS_DIR,
    PROJECT_ROOT,
    TrainConfig,
)
from poker_rl.training import train


def read_json(path: Path) -> Dict[str, Any]:
    with path.open() as input_file:
        return json.load(input_file)


def budget_slug(budget_seconds: float) -> str:
    return f"{budget_seconds:g}".replace(".", "p")


def baseline_run_name(
    algorithm: str,
    seed: int,
    budget_seconds: float,
) -> str:
    return (
        f"baseline-{algorithm}-seed{seed}-"
        f"budget{budget_slug(budget_seconds)}s"
    )


def validate_existing_run(
    run_directory: Path,
    algorithm: str,
    seed: int,
    budget_seconds: float,
) -> None:
    config_path = run_directory / "config.json"
    summary_path = run_directory / "summary.json"

    if not config_path.exists() or not summary_path.exists():
        raise ValueError(
            f"Incomplete existing run: {run_directory}"
        )

    existing_config = read_json(config_path)
    existing_summary = read_json(summary_path)

    expected = {
        "algorithm": algorithm,
        "seed": seed,
        "budget_seconds": budget_seconds,
    }

    actual = {
        key: existing_config.get(key)
        for key in expected
    }

    if actual != expected:
        raise ValueError(
            f"Existing run configuration mismatch in "
            f"{run_directory}: expected {expected}, "
            f"found {actual}."
        )

    if existing_summary.get("status") != "complete":
        raise ValueError(
            f"Existing run did not complete: "
            f"{run_directory}"
        )


def run_training_suite(
    algorithms: List[str],
    seeds: List[int],
    budget_seconds: float,
    log_every: int,
    output_path: Path,
) -> List[Dict[str, Any]]:
    if not algorithms:
        raise ValueError(
            "At least one algorithm is required."
        )

    if not seeds:
        raise ValueError(
            "At least one training seed is required."
        )

    rows = []

    for seed in seeds:
        for algorithm in algorithms:
            run_name = baseline_run_name(
                algorithm,
                seed,
                budget_seconds,
            )
            run_directory = ARTIFACTS_DIR / run_name

            if run_directory.exists():
                validate_existing_run(
                    run_directory,
                    algorithm,
                    seed,
                    budget_seconds,
                )
                action = "reused"
                print(
                    f"Reusing completed run: "
                    f"{run_directory}"
                )

            else:
                config = TrainConfig(
                    algorithm=algorithm,
                    seed=seed,
                    budget_seconds=budget_seconds,
                    log_every=log_every,
                    run_name=run_name,
                )

                run_directory = train(config)
                action = "trained"

            summary = read_json(
                run_directory / "summary.json"
            )

            rows.append(
                {
                    "algorithm": algorithm,
                    "training_seed": seed,
                    "budget_seconds": budget_seconds,
                    "completed_units": summary[
                        "completed_units"
                    ],
                    "unit_name": summary["unit_name"],
                    "training_elapsed_seconds": summary[
                        "training_elapsed_seconds"
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

    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "algorithm",
                "training_seed",
                "budget_seconds",
                "completed_units",
                "unit_name",
                "training_elapsed_seconds",
                "run_directory",
                "action",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Saved training manifest to {output_path}"
    )

    return rows