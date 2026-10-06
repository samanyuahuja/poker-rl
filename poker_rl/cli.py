"""Command-line interface for reproducible poker experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional
from poker_rl.benchmark import run_baseline_benchmark
from poker_rl.config import TrainConfig
from poker_rl.evaluation import evaluate_matchup
from poker_rl.pool_config import PoolTrainConfig
from poker_rl.pool_training import train_with_opponent_pool
from poker_rl.suite import run_training_suite
from poker_rl.training import train
from poker_rl.pool_suite import run_pool_training_suite
from poker_rl.pool_benchmark import run_pool_benchmark

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m poker_rl",
        description=(
            "Train and evaluate reproducible reinforcement-learning "
            "experiments in Leduc Hold'em."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train_parser = commands.add_parser(
        "train",
        help="Train one agent using a fixed time budget.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    train_parser.add_argument(
        "--algorithm",
        choices=["dqn", "nfsp", "cfr"],
        required=True,
        help="Training algorithm.",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training random seed.",
    )
    train_parser.add_argument(
        "--budget-seconds",
        type=float,
        required=True,
        help="Maximum wall-clock training time.",
    )
    train_parser.add_argument(
        "--max-units",
        type=int,
        default=None,
        help="Optional maximum episodes or CFR iterations.",
    )
    train_parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Progress logging interval.",
    )
    train_parser.add_argument(
        "--run-name",
        default=None,
        help="Optional artifact directory name.",
    )

    evaluate_parser = commands.add_parser(
        "evaluate",
        help="Evaluate two saved runs across both seats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    evaluate_parser.add_argument(
        "--agent-a",
        required=True,
        help=(
            "Run directory for the measured agent, "
            "or 'random'."
        ),
    )
    evaluate_parser.add_argument(
        "--agent-b",
        required=True,
        help="Opponent run directory, or 'random'.",
    )
    evaluate_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[11, 22, 33, 44, 55],
        help="Evaluation seeds.",
    )
    evaluate_parser.add_argument(
        "--games-per-seat",
        type=int,
        default=2000,
        help="Games played from each seat for every seed.",
    )
    evaluate_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV path.",
    )

    suite_parser = commands.add_parser(
        "suite",
        help=(
            "Train several algorithms and seeds using the same "
            "wall-clock budget."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    suite_parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["dqn", "nfsp", "cfr"],
        default=["dqn", "nfsp", "cfr"],
        help="Algorithms included in the suite.",
    )
    suite_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Independent training seeds.",
    )
    suite_parser.add_argument(
        "--budget-seconds",
        type=float,
        required=True,
        help="Training time given to every algorithm and seed.",
    )
    suite_parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Progress logging interval.",
    )
    suite_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination training-manifest CSV.",
    )
    benchmark_parser = commands.add_parser(
        "benchmark",
        help=(
            "Evaluate all matched-budget baseline matchups."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    benchmark_parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Training-manifest CSV created by suite.",
    )
    benchmark_parser.add_argument(
        "--evaluation-seeds",
        type=int,
        nargs="+",
        default=[11, 22, 33, 44, 55],
        help="Independent evaluation seeds.",
    )
    benchmark_parser.add_argument(
        "--games-per-seat",
        type=int,
        default=2000,
        help="Games played from each seat.",
    )
    benchmark_parser.add_argument(
        "--raw-output",
        type=Path,
        required=True,
        help="Destination for detailed evaluation results.",
    )
    benchmark_parser.add_argument(
        "--summary-output",
        type=Path,
        required=True,
        help="Destination for summarized results.",
    )
    pool_train_parser = commands.add_parser(
        "pool-train",
        help="Train DQN against a historical opponent pool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    pool_train_parser.add_argument(
        "--strategy",
        choices=[
            "latest",
            "uniform",
            "pfsp",
            "uncertainty",
        ],
        required=True,
        help="Historical-opponent selection rule.",
    )
    pool_train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training random seed.",
    )
    pool_train_parser.add_argument(
        "--budget-seconds",
        type=float,
        required=True,
        help="Total wall-clock training budget.",
    )
    pool_train_parser.add_argument(
        "--checkpoint-seconds",
        type=float,
        default=30,
        help="Seconds between historical checkpoints.",
    )
    pool_train_parser.add_argument(
        "--evaluation-games",
        type=int,
        default=100,
        help="Seat-balanced games per pool opponent.",
    )
    pool_train_parser.add_argument(
        "--max-pool-size",
        type=int,
        default=10,
        help="Maximum active historical opponents.",
    )
    pool_train_parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Uncertainty-bonus strength.",
    )
    pool_train_parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Uniform sampling mixture.",
    )
    pool_train_parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Uncertainty-strategy softmax temperature.",
    )
    pool_train_parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Episode logging interval.",
    )
    pool_train_parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional episode limit for smoke tests.",
    )
    pool_train_parser.add_argument(
        "--run-name",
        default=None,
        help="Optional artifact directory name.",
    )
    pool_suite_parser = commands.add_parser(
        "pool-suite",
        help=(
            "Train historical opponent-pool strategies "
            "using matched wall-clock budgets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    pool_suite_parser.add_argument(
        "--strategies",
        nargs="+",
        choices=[
            "latest",
            "uniform",
            "pfsp",
            "uncertainty",
        ],
        default=[
            "latest",
            "uniform",
            "pfsp",
            "uncertainty",
        ],
        help="Opponent-selection strategies.",
    )
    pool_suite_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Independent training seeds.",
    )
    pool_suite_parser.add_argument(
        "--budget-seconds",
        type=float,
        required=True,
        help="Training time given to every run.",
    )
    pool_suite_parser.add_argument(
        "--checkpoint-seconds",
        type=float,
        default=30,
        help="Seconds between historical checkpoints.",
    )
    pool_suite_parser.add_argument(
        "--evaluation-games",
        type=int,
        default=100,
        help="Seat-balanced games per pool opponent.",
    )
    pool_suite_parser.add_argument(
        "--max-pool-size",
        type=int,
        default=10,
        help="Maximum active historical opponents.",
    )
    pool_suite_parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Uncertainty-bonus strength.",
    )
    pool_suite_parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Uniform sampling mixture.",
    )
    pool_suite_parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Uncertainty softmax temperature.",
    )
    pool_suite_parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Episode logging interval.",
    )
    pool_suite_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination training-manifest CSV.",
    )
    pool_benchmark_parser = commands.add_parser(
        "pool-benchmark",
        help=(
            "Evaluate opponent-pool strategies "
            "against baselines and each other."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    pool_benchmark_parser.add_argument(
        "--pool-manifest",
        type=Path,
        required=True,
        help="Manifest created by pool-suite.",
    )
    pool_benchmark_parser.add_argument(
        "--baseline-manifest",
        type=Path,
        required=True,
        help="Matched-budget baseline manifest.",
    )
    pool_benchmark_parser.add_argument(
        "--evaluation-seeds",
        type=int,
        nargs="+",
        default=[11, 22, 33, 44, 55],
        help="Independent evaluation seeds.",
    )
    pool_benchmark_parser.add_argument(
        "--games-per-seat",
        type=int,
        default=2000,
        help="Games played from each player position.",
    )
    pool_benchmark_parser.add_argument(
        "--raw-output",
        type=Path,
        required=True,
        help="Detailed evaluation CSV.",
    )
    pool_benchmark_parser.add_argument(
        "--summary-output",
        type=Path,
        required=True,
        help="Matchup-summary CSV.",
    )
    pool_benchmark_parser.add_argument(
        "--aggregate-output",
        type=Path,
        required=True,
        help="Aggregate research-metrics CSV.",
    )
    return parser


def run_train_command(args: argparse.Namespace) -> int:
    config = TrainConfig(
        algorithm=args.algorithm,
        seed=args.seed,
        budget_seconds=args.budget_seconds,
        max_units=args.max_units,
        log_every=args.log_every,
        run_name=args.run_name,
    )

    run_directory = train(config)
    print(f"Artifacts saved in: {run_directory}")

    return 0


def run_evaluate_command(
    args: argparse.Namespace,
) -> int:
    evaluate_matchup(
        agent_a_spec=args.agent_a,
        agent_b_spec=args.agent_b,
        seeds=args.seeds,
        games_per_seat=args.games_per_seat,
        output_path=args.output,
    )

    return 0


def run_suite_command(args: argparse.Namespace) -> int:
    run_training_suite(
        algorithms=args.algorithms,
        seeds=args.seeds,
        budget_seconds=args.budget_seconds,
        log_every=args.log_every,
        output_path=args.output,
    )

    return 0


def run_benchmark_command(
    args: argparse.Namespace,
) -> int:
    run_baseline_benchmark(
        manifest_path=args.manifest,
        evaluation_seeds=args.evaluation_seeds,
        games_per_seat=args.games_per_seat,
        raw_output_path=args.raw_output,
        summary_output_path=args.summary_output,
    )

    return 0


def run_pool_train_command(
    args: argparse.Namespace,
) -> int:
    config = PoolTrainConfig(
        strategy=args.strategy,
        seed=args.seed,
        budget_seconds=args.budget_seconds,
        checkpoint_seconds=args.checkpoint_seconds,
        evaluation_games=args.evaluation_games,
        max_pool_size=args.max_pool_size,
        beta=args.beta,
        epsilon=args.epsilon,
        temperature=args.temperature,
        log_every=args.log_every,
        max_episodes=args.max_episodes,
        run_name=args.run_name,
    )

    run_directory = train_with_opponent_pool(config)

    print(f"Artifacts saved in: {run_directory}")
    return 0


def run_pool_suite_command(
    args: argparse.Namespace,
) -> int:
    run_pool_training_suite(
        strategies=args.strategies,
        seeds=args.seeds,
        budget_seconds=args.budget_seconds,
        checkpoint_seconds=args.checkpoint_seconds,
        evaluation_games=args.evaluation_games,
        max_pool_size=args.max_pool_size,
        beta=args.beta,
        epsilon=args.epsilon,
        temperature=args.temperature,
        log_every=args.log_every,
        output_path=args.output,
    )

    return 0


def run_pool_benchmark_command(
    args: argparse.Namespace,
) -> int:
    run_pool_benchmark(
        pool_manifest_path=(
            args.pool_manifest
        ),
        baseline_manifest_path=(
            args.baseline_manifest
        ),
        evaluation_seeds=(
            args.evaluation_seeds
        ),
        games_per_seat=(
            args.games_per_seat
        ),
        raw_output_path=args.raw_output,
        summary_output_path=(
            args.summary_output
        ),
        aggregate_output_path=(
            args.aggregate_output
        ),
    )

    return 0


def main(arguments: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(arguments)

    if args.command == "train":
        return run_train_command(args)

    if args.command == "evaluate":
        return run_evaluate_command(args)

    if args.command == "suite":
        return run_suite_command(args)

    if args.command == "benchmark":
        return run_benchmark_command(args)

    if args.command == "pool-train":
        return run_pool_train_command(args)

    if args.command == "pool-suite":
        return run_pool_suite_command(args)

    if args.command == "pool-benchmark":
        return run_pool_benchmark_command(args)

    parser.error(f"Unknown command: {args.command}")
    return 2
