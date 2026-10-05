"""Configuration for historical opponent-pool training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from poker_rl.config import ARTIFACTS_DIR
from poker_rl.opponent_selection import (
    SUPPORTED_SELECTION_STRATEGIES,
)


@dataclass(frozen=True)
class PoolTrainConfig:
    strategy: str
    seed: int
    budget_seconds: float
    checkpoint_seconds: float = 30
    evaluation_games: int = 100
    max_pool_size: int = 10
    beta: float = 1.0
    epsilon: float = 0.1
    temperature: float = 1.0
    log_every: int = 1000
    max_episodes: Optional[int] = None
    run_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.strategy not in SUPPORTED_SELECTION_STRATEGIES:
            choices = ", ".join(
                sorted(SUPPORTED_SELECTION_STRATEGIES)
            )
            raise ValueError(
                f"Unknown strategy '{self.strategy}'. "
                f"Choose from: {choices}."
            )

        if self.budget_seconds <= 0:
            raise ValueError(
                "budget_seconds must be greater than zero."
            )

        if self.checkpoint_seconds <= 0:
            raise ValueError(
                "checkpoint_seconds must be greater than zero."
            )

        if self.checkpoint_seconds > self.budget_seconds:
            raise ValueError(
                "checkpoint_seconds cannot exceed "
                "budget_seconds."
            )

        if (
            self.evaluation_games <= 0
            or self.evaluation_games % 2 != 0
        ):
            raise ValueError(
                "evaluation_games must be a positive even number."
            )

        if self.max_pool_size <= 0:
            raise ValueError(
                "max_pool_size must be greater than zero."
            )

        if self.beta < 0:
            raise ValueError("beta cannot be negative.")

        if not 0 <= self.epsilon <= 1:
            raise ValueError(
                "epsilon must be between zero and one."
            )

        if self.temperature <= 0:
            raise ValueError(
                "temperature must be greater than zero."
            )

        if self.log_every <= 0:
            raise ValueError(
                "log_every must be greater than zero."
            )

        if (
            self.max_episodes is not None
            and self.max_episodes <= 0
        ):
            raise ValueError(
                "max_episodes must be greater than zero."
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


def pool_config_as_dict(
    config: PoolTrainConfig,
) -> Dict[str, object]:
    data = asdict(config)
    data["algorithm"] = "dqn"
    data["experiment"] = "historical_opponent_pool"
    return data


def create_pool_run_directory(
    config: PoolTrainConfig,
) -> Path:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    run_name = (
        config.run_name
        or (
            f"pool-{config.strategy}-seed{config.seed}-"
            f"{timestamp}"
        )
    )

    run_directory = ARTIFACTS_DIR / run_name
    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    return run_directory