import csv
import math
import statistics
from pathlib import Path

import rlcard
import torch

from rlcard.agents import RandomAgent
from rlcard.utils import set_seed, tournament

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEEDS = [11, 22, 33, 44, 55]
GAMES_PER_MATCHUP = 2000

dqn = torch.load(
    PROJECT_DIR / "poker_dqn.pth",
    map_location="cpu",
    weights_only=False,
)

nfsp_player_0 = torch.load(
    PROJECT_DIR / "nfsp_player_0_50k.pth",
    map_location="cpu",
    weights_only=False,
)

nfsp_player_1 = torch.load(
    PROJECT_DIR / "nfsp_player_1_50k.pth",
    map_location="cpu",
    weights_only=False,
)

matchups = [
    ("dqn_vs_random_seat_0", "dqn", "random", 0),
    ("dqn_vs_random_seat_1", "random", "dqn", 1),
    ("nfsp_vs_random_seat_0", "nfsp_0", "random", 0),
    ("nfsp_vs_random_seat_1", "random", "nfsp_1", 1),
    ("dqn_vs_nfsp", "dqn", "nfsp_1", 0),
    ("nfsp_vs_dqn", "nfsp_0", "dqn", 0),
]

results = []

for seed in SEEDS:
    for matchup_name, seat_0_name, seat_1_name, measured_seat in matchups:
        set_seed(seed)

        env = rlcard.make("leduc-holdem", config={"seed": seed})
        random_agent = RandomAgent(num_actions=env.num_actions)

        agents = {
            "dqn": dqn,
            "nfsp_0": nfsp_player_0,
            "nfsp_1": nfsp_player_1,
            "random": random_agent,
        }

        env.set_agents([agents[seat_0_name], agents[seat_1_name]])
        payoffs = tournament(env, GAMES_PER_MATCHUP)
        score = float(payoffs[measured_seat])

        results.append(
            {
                "matchup": matchup_name,
                "seed": seed,
                "games": GAMES_PER_MATCHUP,
                "average_payoff": score,
            }
        )

        print(f"{matchup_name}, seed {seed}: {score:.4f}")

csv_path = RESULTS_DIR / "baseline_evaluation.csv"

with csv_path.open("w", newline="") as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=["matchup", "seed", "games", "average_payoff"],
    )
    writer.writeheader()
    writer.writerows(results)

print("\nSummary with 95% confidence intervals:")

for matchup_name, _, _, _ in matchups:
    scores = [
        row["average_payoff"]
        for row in results
        if row["matchup"] == matchup_name
    ]

    mean_score = statistics.mean(scores)
    standard_error = statistics.stdev(scores) / math.sqrt(len(scores))
    confidence_interval = 2.776 * standard_error

    print(
        f"{matchup_name}: "
        f"{mean_score:.4f} ± {confidence_interval:.4f}"
    )

print(f"\nSaved detailed results to {csv_path}")