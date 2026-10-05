"""Tests for matched-budget opponent-pool suites."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import poker_rl.pool_suite as pool_suite
from poker_rl.pool_config import (
    PoolTrainConfig,
    pool_config_as_dict,
)
from poker_rl.pool_suite import (
    pool_run_name,
    run_pool_training_suite,
    validate_existing_pool_run,
)


class PoolSuiteTests(unittest.TestCase):
    def create_complete_run(
        self,
        run_directory: Path,
        config: PoolTrainConfig,
    ) -> None:
        run_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        with (
            run_directory / "config.json"
        ).open("w") as output_file:
            json.dump(
                pool_config_as_dict(config),
                output_file,
            )

        with (
            run_directory / "summary.json"
        ).open("w") as output_file:
            json.dump(
                {
                    "status": "complete",
                    "completed_units": 100,
                    "training_elapsed_seconds": 2.0,
                    "post_training_evaluation_seconds": 0.1,
                    "historical_checkpoint_count": 4,
                    "final_pool_size": 3,
                },
                output_file,
            )

        for filename in [
            "player_0.pth",
            "player_1.pth",
            "training_progress.csv",
            "pool_statistics.csv",
        ]:
            (run_directory / filename).touch()

    def test_run_name_is_deterministic(self):
        self.assertEqual(
            pool_run_name(
                strategy="uncertainty",
                seed=3,
                budget_seconds=300,
            ),
            "pool-uncertainty-seed3-budget300s",
        )

    def test_validates_complete_matching_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory) / "run"

            config = PoolTrainConfig(
                strategy="pfsp",
                seed=2,
                budget_seconds=300,
                run_name="run",
            )

            self.create_complete_run(
                run_directory,
                config,
            )

            validate_existing_pool_run(
                run_directory,
                config,
            )

    def test_rejects_configuration_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory) / "run"

            saved_config = PoolTrainConfig(
                strategy="uniform",
                seed=1,
                budget_seconds=300,
                run_name="run",
            )

            requested_config = PoolTrainConfig(
                strategy="uncertainty",
                seed=1,
                budget_seconds=300,
                run_name="run",
            )

            self.create_complete_run(
                run_directory,
                saved_config,
            )

            with self.assertRaises(ValueError):
                validate_existing_pool_run(
                    run_directory,
                    requested_config,
                )

    def test_rejects_duplicate_strategies(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_pool_training_suite(
                    strategies=["pfsp", "pfsp"],
                    seeds=[1],
                    budget_seconds=300,
                    checkpoint_seconds=30,
                    evaluation_games=100,
                    max_pool_size=10,
                    beta=1.0,
                    epsilon=0.1,
                    temperature=1.0,
                    log_every=1000,
                    output_path=(
                        Path(directory) / "manifest.csv"
                    ),
                )

    def test_rejects_duplicate_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_pool_training_suite(
                    strategies=["latest"],
                    seeds=[1, 1],
                    budget_seconds=300,
                    checkpoint_seconds=30,
                    evaluation_games=100,
                    max_pool_size=10,
                    beta=1.0,
                    epsilon=0.1,
                    temperature=1.0,
                    log_every=1000,
                    output_path=(
                        Path(directory) / "manifest.csv"
                    ),
                )

    def test_reuses_complete_run_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            artifact_directory = (
                project_root / "artifacts"
            )
            artifact_directory.mkdir()

            run_name = pool_run_name(
                strategy="latest",
                seed=1,
                budget_seconds=300,
            )

            config = PoolTrainConfig(
                strategy="latest",
                seed=1,
                budget_seconds=300,
                checkpoint_seconds=30,
                evaluation_games=100,
                max_pool_size=10,
                beta=1.0,
                epsilon=0.1,
                temperature=1.0,
                log_every=1000,
                run_name=run_name,
            )

            self.create_complete_run(
                artifact_directory / run_name,
                config,
            )

            output_path = (
                project_root / "manifest.csv"
            )

            with (
                mock.patch.object(
                    pool_suite,
                    "PROJECT_ROOT",
                    project_root,
                ),
                mock.patch.object(
                    pool_suite,
                    "ARTIFACTS_DIR",
                    artifact_directory,
                ),
            ):
                rows = run_pool_training_suite(
                    strategies=["latest"],
                    seeds=[1],
                    budget_seconds=300,
                    checkpoint_seconds=30,
                    evaluation_games=100,
                    max_pool_size=10,
                    beta=1.0,
                    epsilon=0.1,
                    temperature=1.0,
                    log_every=1000,
                    output_path=output_path,
                )

            self.assertEqual(
                rows[0]["action"],
                "reused",
            )
            self.assertEqual(
                rows[0]["completed_episodes"],
                100,
            )
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()