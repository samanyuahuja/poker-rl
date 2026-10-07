"""Tests for matched-budget ablation training."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import poker_rl.ablation_suite as ablation_suite
from poker_rl.ablation_suite import (
    ABLATION_BY_LABEL,
    ablation_run_name,
    run_ablation_suite,
    validate_labels,
)
from poker_rl.pool_config import (
    PoolTrainConfig,
    pool_config_as_dict,
)


class AblationSuiteTests(unittest.TestCase):
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

    def test_defines_controlled_ablation_families(self):
        self.assertEqual(
            ABLATION_BY_LABEL[
                "beta-0"
            ].max_pool_size,
            10,
        )
        self.assertEqual(
            ABLATION_BY_LABEL[
                "beta-2"
            ].checkpoint_seconds,
            30,
        )
        self.assertEqual(
            ABLATION_BY_LABEL[
                "pool-5"
            ].checkpoint_seconds,
            10,
        )
        self.assertEqual(
            ABLATION_BY_LABEL[
                "pool-20"
            ].max_pool_size,
            20,
        )

    def test_run_name_is_deterministic(self):
        self.assertEqual(
            ablation_run_name(
                label="beta-0",
                seed=3,
                budget_seconds=300,
            ),
            (
                "ablation-beta-0-seed3-"
                "budget300s"
            ),
        )

    def test_rejects_unknown_label(self):
        with self.assertRaises(ValueError):
            validate_labels(
                ["unknown"]
            )

    def test_rejects_duplicate_labels(self):
        with self.assertRaises(ValueError):
            validate_labels(
                ["beta-0", "beta-0"]
            )

    def test_reuses_complete_run_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as raw:
            project_root = Path(raw)
            artifact_directory = (
                project_root / "artifacts"
            )
            artifact_directory.mkdir()

            run_name = ablation_run_name(
                label="beta-0",
                seed=1,
                budget_seconds=300,
            )

            config = PoolTrainConfig(
                strategy="uncertainty",
                seed=1,
                budget_seconds=300,
                checkpoint_seconds=30,
                evaluation_games=100,
                max_pool_size=10,
                beta=0,
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
                project_root
                / "ablation_manifest.csv"
            )

            with (
                mock.patch.object(
                    ablation_suite,
                    "PROJECT_ROOT",
                    project_root,
                ),
                mock.patch.object(
                    ablation_suite,
                    "ARTIFACTS_DIR",
                    artifact_directory,
                ),
            ):
                rows = run_ablation_suite(
                    labels=["beta-0"],
                    seeds=[1],
                    budget_seconds=300,
                    evaluation_games=100,
                    epsilon=0.1,
                    temperature=1.0,
                    log_every=1000,
                    output_path=output_path,
                )

            self.assertEqual(
                len(rows),
                1,
            )
            self.assertEqual(
                rows[0]["action"],
                "reused",
            )
            self.assertEqual(
                rows[0]["ablation_family"],
                "uncertainty_strength",
            )
            self.assertEqual(
                rows[0]["beta"],
                0,
            )
            self.assertTrue(
                output_path.exists()
            )


if __name__ == "__main__":
    unittest.main()