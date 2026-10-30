"""Play terminal Leduc Hold'em against a saved experiment run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import rlcard

from rlcard.agents.human_agents.leduc_holdem_human_agent import (
    HumanAgent,
)

from poker_rl.evaluation import load_agent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Play Leduc Hold'em against a trained poker_rl agent."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agent",
        type=Path,
        default=Path("artifacts/demo-dqn"),
        help="Completed training-run directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Game environment seed.",
    )
    return parser


def play(agent_directory: Path, seed: int) -> None:
    environment = rlcard.make(
        "leduc-holdem",
        config={"seed": seed},
    )
    learning_agent = load_agent(
        spec=agent_directory,
        seat=1,
        env=environment,
    )
    human_agent = HumanAgent(
        num_actions=environment.num_actions
    )
    environment.set_agents(
        [human_agent, learning_agent]
    )

    while True:
        _, payoffs = environment.run(
            is_training=False
        )

        print("\nGame finished!")
        print("Your payoff:", payoffs[0])
        print("AI payoff:", payoffs[1])

        play_again = input(
            "\nPlay again? Enter y or n: "
        ).strip().lower()

        if play_again != "y":
            print("Thanks for playing!")
            return


def main(arguments: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(arguments)
    play(
        agent_directory=args.agent,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
