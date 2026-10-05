"""Historical checkpoint pool for self-play training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from poker_rl.opponent_selection import (
    SUPPORTED_SELECTION_STRATEGIES,
    select_opponent,
    selection_probabilities,
)


@dataclass
class OpponentRecord:
    checkpoint_path: Path
    created_episode: int
    learner_wins: int = 0
    learner_losses: int = 0
    ties: int = 0
    selection_count: int = 0

    @property
    def evaluation_count(self) -> int:
        return (
            self.learner_wins
            + self.learner_losses
            + self.ties
        )

    @property
    def learner_loss_rate(self) -> float:
        if self.evaluation_count == 0:
            return 0.5

        opponent_score = (
            self.learner_losses
            + 0.5 * self.ties
        )

        return opponent_score / self.evaluation_count

    def record_payoff(self, learner_payoff: float) -> None:
        if learner_payoff > 0:
            self.learner_wins += 1
        elif learner_payoff < 0:
            self.learner_losses += 1
        else:
            self.ties += 1


class HistoricalOpponentPool:
    """Store checkpoints and sample training opponents."""

    def __init__(
        self,
        strategy: str,
        seed: int,
        max_size: int = 10,
        beta: float = 1.0,
        epsilon: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        if strategy not in SUPPORTED_SELECTION_STRATEGIES:
            choices = ", ".join(
                sorted(SUPPORTED_SELECTION_STRATEGIES)
            )
            raise ValueError(
                f"Unknown selection strategy '{strategy}'. "
                f"Choose from: {choices}."
            )

        if max_size <= 0:
            raise ValueError(
                "max_size must be greater than zero."
            )

        self.strategy = strategy
        self.max_size = max_size
        self.beta = beta
        self.epsilon = epsilon
        self.temperature = temperature
        self.random_generator = np.random.default_rng(seed)
        self.records: List[OpponentRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def add_checkpoint(
        self,
        checkpoint_path: Path,
        created_episode: int,
    ) -> Optional[OpponentRecord]:
        resolved_path = checkpoint_path.resolve()

        if any(
            record.checkpoint_path == resolved_path
            for record in self.records
        ):
            raise ValueError(
                f"Checkpoint is already in the pool: "
                f"{resolved_path}"
            )

        removed_record = None

        if len(self.records) >= self.max_size:
            removed_record = self.records.pop(0)

        self.records.append(
            OpponentRecord(
                checkpoint_path=resolved_path,
                created_episode=created_episode,
            )
        )

        return removed_record

    def probabilities(self) -> np.ndarray:
        if not self.records:
            raise ValueError(
                "Cannot sample from an empty opponent pool."
            )

        return selection_probabilities(
            strategy=self.strategy,
            loss_rates=[
                record.learner_loss_rate
                for record in self.records
            ],
            evaluation_counts=[
                record.evaluation_count
                for record in self.records
            ],
            beta=self.beta,
            epsilon=self.epsilon,
            temperature=self.temperature,
        )

    def sample(self) -> OpponentRecord:
        probabilities = self.probabilities()

        opponent_index = select_opponent(
            probabilities,
            self.random_generator,
        )

        record = self.records[opponent_index]
        record.selection_count += 1
        return record

    def record_payoff(
        self,
        checkpoint_path: Path,
        learner_payoff: float,
    ) -> None:
        resolved_path = checkpoint_path.resolve()

        for record in self.records:
            if record.checkpoint_path == resolved_path:
                record.record_payoff(learner_payoff)
                return

        raise ValueError(
            f"Checkpoint is not in the pool: "
            f"{resolved_path}"
        )

    def rows(self) -> List[Dict[str, object]]:
        probabilities = self.probabilities()

        return [
            {
                "checkpoint_path": str(
                    record.checkpoint_path
                ),
                "created_episode": (
                    record.created_episode
                ),
                "learner_wins": record.learner_wins,
                "learner_losses": record.learner_losses,
                "ties": record.ties,
                "evaluation_count": (
                    record.evaluation_count
                ),
                "learner_loss_rate": (
                    record.learner_loss_rate
                ),
                "selection_count": (
                    record.selection_count
                ),
                "selection_probability": float(
                    probabilities[index]
                ),
            }
            for index, record in enumerate(self.records)
        ]