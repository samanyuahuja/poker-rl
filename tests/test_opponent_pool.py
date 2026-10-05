"""Tests for the historical opponent pool."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from poker_rl.opponent_pool import (
    HistoricalOpponentPool,
    OpponentRecord,
)


class OpponentRecordTests(unittest.TestCase):
    def test_tracks_payoffs_and_loss_rate(self):
        record = OpponentRecord(
            checkpoint_path=Path("checkpoint.pth"),
            created_episode=100,
        )

        self.assertEqual(record.learner_loss_rate, 0.5)

        record.record_payoff(1.0)
        record.record_payoff(-1.0)
        record.record_payoff(0.0)

        self.assertEqual(record.evaluation_count, 3)
        self.assertEqual(record.learner_wins, 1)
        self.assertEqual(record.learner_losses, 1)
        self.assertEqual(record.ties, 1)
        self.assertEqual(record.learner_loss_rate, 0.5)


class HistoricalOpponentPoolTests(unittest.TestCase):
    def test_evicts_oldest_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)

            pool = HistoricalOpponentPool(
                strategy="uniform",
                seed=1,
                max_size=2,
            )

            first = root / "first.pth"
            second = root / "second.pth"
            third = root / "third.pth"

            pool.add_checkpoint(first, 10)
            pool.add_checkpoint(second, 20)
            removed = pool.add_checkpoint(third, 30)

            self.assertIsNotNone(removed)
            self.assertEqual(
                removed.checkpoint_path,
                first.resolve(),
            )
            self.assertEqual(len(pool), 2)
            self.assertEqual(
                pool.records[0].checkpoint_path,
                second.resolve(),
            )
            self.assertEqual(
                pool.records[1].checkpoint_path,
                third.resolve(),
            )

    def test_latest_always_samples_newest_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)

            pool = HistoricalOpponentPool(
                strategy="latest",
                seed=1,
            )

            pool.add_checkpoint(root / "old.pth", 10)
            pool.add_checkpoint(root / "new.pth", 20)

            samples = [
                pool.sample().created_episode
                for _ in range(10)
            ]

            self.assertEqual(samples, [20] * 10)
            self.assertEqual(
                pool.records[1].selection_count,
                10,
            )

    def test_pfsp_prioritizes_checkpoint_causing_losses(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)

            pool = HistoricalOpponentPool(
                strategy="pfsp",
                seed=1,
                epsilon=0,
            )

            difficult = root / "difficult.pth"
            easy = root / "easy.pth"

            pool.add_checkpoint(difficult, 10)
            pool.add_checkpoint(easy, 20)

            for _ in range(10):
                pool.record_payoff(difficult, -1.0)
                pool.record_payoff(easy, 1.0)

            probabilities = pool.probabilities()

            self.assertGreater(
                probabilities[0],
                probabilities[1],
            )

    def test_rows_include_evaluation_statistics(self):
        with TemporaryDirectory() as directory:
            checkpoint = (
                Path(directory) / "checkpoint.pth"
            )

            pool = HistoricalOpponentPool(
                strategy="uniform",
                seed=1,
            )
            pool.add_checkpoint(checkpoint, 50)
            pool.record_payoff(checkpoint, -2.0)
            pool.sample()

            row = pool.rows()[0]

            self.assertEqual(row["created_episode"], 50)
            self.assertEqual(row["learner_losses"], 1)
            self.assertEqual(row["evaluation_count"], 1)
            self.assertEqual(row["selection_count"], 1)
            self.assertEqual(
                row["learner_loss_rate"],
                1.0,
            )

    def test_rejects_duplicate_checkpoint(self):
        with TemporaryDirectory() as directory:
            checkpoint = (
                Path(directory) / "checkpoint.pth"
            )

            pool = HistoricalOpponentPool(
                strategy="uniform",
                seed=1,
            )
            pool.add_checkpoint(checkpoint, 10)

            with self.assertRaises(ValueError):
                pool.add_checkpoint(checkpoint, 20)


if __name__ == "__main__":
    unittest.main()