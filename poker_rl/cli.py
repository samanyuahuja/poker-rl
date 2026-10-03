"""Command-line interface for reproducible poker experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from poker_rl.config import TrainConfig
from poker_rl.evaluation import evaluate_matchup
from poker_rl.suite import run_training_suite
from poker_rl.training import train
from poker_rl.benchmark import run_baseline_benchmark


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

    parser.error(f"Unknown command: {args.command}")
    return 2
