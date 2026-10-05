"""Opponent-selection rules for historical self-play."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

SUPPORTED_SELECTION_STRATEGIES = {
    "latest",
    "uniform",
    "pfsp",
    "uncertainty",
}


def validate_inputs(
    strategy: str,
    loss_rates: np.ndarray,
    evaluation_counts: np.ndarray,
    beta: float,
    epsilon: float,
    temperature: float,
) -> None:
    if strategy not in SUPPORTED_SELECTION_STRATEGIES:
        choices = ", ".join(
            sorted(SUPPORTED_SELECTION_STRATEGIES)
        )
        raise ValueError(
            f"Unknown selection strategy '{strategy}'. "
            f"Choose from: {choices}."
        )

    if loss_rates.ndim != 1:
        raise ValueError("loss_rates must be one-dimensional.")

    if evaluation_counts.shape != loss_rates.shape:
        raise ValueError(
            "evaluation_counts must match loss_rates."
        )

    if len(loss_rates) == 0:
        raise ValueError(
            "At least one opponent is required."
        )

    if np.any(loss_rates < 0) or np.any(loss_rates > 1):
        raise ValueError(
            "Every loss rate must be between zero and one."
        )

    if np.any(evaluation_counts < 0):
        raise ValueError(
            "Evaluation counts cannot be negative."
        )

    if beta < 0:
        raise ValueError("beta cannot be negative.")

    if not 0 <= epsilon <= 1:
        raise ValueError(
            "epsilon must be between zero and one."
        )

    if temperature <= 0:
        raise ValueError(
            "temperature must be greater than zero."
        )


def normalized_scores(scores: np.ndarray) -> np.ndarray:
    total = float(scores.sum())

    if total <= 0:
        return np.full(
            len(scores),
            1.0 / len(scores),
        )

    return scores / total


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def selection_probabilities(
    strategy: str,
    loss_rates: Sequence[float],
    evaluation_counts: Sequence[int],
    beta: float = 1.0,
    epsilon: float = 0.1,
    temperature: float = 1.0,
) -> np.ndarray:
    """Return one probability for every historical opponent."""

    losses = np.asarray(loss_rates, dtype=float)
    counts = np.asarray(
        evaluation_counts,
        dtype=float,
    )

    validate_inputs(
        strategy=strategy,
        loss_rates=losses,
        evaluation_counts=counts,
        beta=beta,
        epsilon=epsilon,
        temperature=temperature,
    )

    opponent_count = len(losses)
    uniform = np.full(
        opponent_count,
        1.0 / opponent_count,
    )

    if strategy == "latest":
        probabilities = np.zeros(opponent_count)
        probabilities[-1] = 1.0
        return probabilities

    if strategy == "uniform":
        return uniform

    if strategy == "pfsp":
        base_probabilities = normalized_scores(
            np.square(losses)
        )

    else:
        total_evaluations = float(counts.sum())

        uncertainty_bonus = np.sqrt(
            2
            * math.log(total_evaluations + 2)
            / (counts + 1)
        )

        priority = (
            losses + beta * uncertainty_bonus
        )

        base_probabilities = softmax(
            priority / temperature
        )

    return (
        (1 - epsilon) * base_probabilities
        + epsilon * uniform
    )


def select_opponent(
    probabilities: Sequence[float],
    random_generator: np.random.Generator,
) -> int:
    """Sample and return an opponent index."""

    probability_array = np.asarray(
        probabilities,
        dtype=float,
    )

    if probability_array.ndim != 1:
        raise ValueError(
            "probabilities must be one-dimensional."
        )

    if len(probability_array) == 0:
        raise ValueError(
            "At least one probability is required."
        )

    if np.any(probability_array < 0):
        raise ValueError(
            "Probabilities cannot be negative."
        )

    if not np.isclose(
        probability_array.sum(),
        1.0,
    ):
        raise ValueError(
            "Probabilities must sum to one."
        )

    return int(
        random_generator.choice(
            len(probability_array),
            p=probability_array,
        )
    )