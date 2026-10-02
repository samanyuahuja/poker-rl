"""Configuration and artifact paths shared by all experiments."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

SUPPORTED_ALGORITHMS = {"dqn", "nfsp", "cfr"}


@dataclass(frozen=True)
class TrainConfig:
    algorithm: str
    seed: int
    budget_seconds: float
    max_units: Optional[int] = None
    log_every: int = 100
    run_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.algorithm not in SUPPORTED_ALGORITHMS:
            choices = ", ".join(sorted(SUPPORTED_ALGORITHMS))
            raise ValueError(
                f"Unknown algorithm '{self.algorithm}'. "
                f"Choose from: {choices}."
            )

        if self.budget_seconds <= 0:
            raise ValueError(
                "budget_seconds must be greater than zero."
            )

        if self.max_units is not None and self.max_units <= 0:
            raise ValueError(
                "max_units must be greater than zero."
            )

        if self.log_every <= 0:
            raise ValueError(
                "log_every must be greater than zero."
            )

        if self.run_name is not None:
            run_path = Path(self.run_name)

            if (
                run_path.name != self.run_name
                or self.run_name in {".", ".."}
            ):
                raise ValueError(
                    "run_name must be one directory name "
                    "without path separators."
                )


def create_run_directory(config: TrainConfig) -> Path:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    run_name = (
        config.run_name
        or f"{config.algorithm}-seed{config.seed}-{timestamp}"
    )

    run_directory = ARTIFACTS_DIR / run_name
    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    return run_directory


def run_git_command(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

    return result.stdout.strip()


def environment_metadata() -> Dict[str, Any]:
    git_status = run_git_command(
        "status",
        "--porcelain",
    )

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": run_git_command(
            "rev-parse",
            "HEAD",
        ),
        "git_dirty": (
            "unknown"
            if git_status == "unknown"
            else bool(git_status)
        ),
    }


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w") as output_file:
        json.dump(
            data,
            output_file,
            indent=2,
            sort_keys=True,
        )


def config_as_dict(
    config: TrainConfig,
) -> Dict[str, Any]:
    return asdict(config)