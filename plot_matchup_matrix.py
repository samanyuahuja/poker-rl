import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
OUTPUT_PATH = RESULTS_DIR / "head_to_head_matrix.png"


def read_results(path):
    results = defaultdict(dict)

    with path.open() as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            results[row["matchup"]][int(row["seed"])] = float(
                row["average_payoff"]
            )

    return results


def combined_mean(
    results,
    first_matchup,
    second_matchup,
    first_sign=1,
    second_sign=1,
):
    seeds = sorted(
        set(results[first_matchup])
        & set(results[second_matchup])
    )

    paired_scores = [
        (
            first_sign * results[first_matchup][seed]
            + second_sign * results[second_matchup][seed]
        )
        / 2
        for seed in seeds
    ]

    return statistics.mean(paired_scores)


baseline = read_results(
    RESULTS_DIR / "baseline_evaluation.csv"
)

cfr_results = read_results(
    RESULTS_DIR / "cfr_evaluation.csv"
)

dqn_vs_random = combined_mean(
    baseline,
    "dqn_vs_random_seat_0",
    "dqn_vs_random_seat_1",
)

nfsp_vs_random = combined_mean(
    baseline,
    "nfsp_vs_random_seat_0",
    "nfsp_vs_random_seat_1",
)

dqn_vs_nfsp = combined_mean(
    baseline,
    "dqn_vs_nfsp",
    "nfsp_vs_dqn",
    first_sign=1,
    second_sign=-1,
)

cfr_vs_random = combined_mean(
    cfr_results,
    "cfr_vs_random_seat_0",
    "cfr_vs_random_seat_1",
)

cfr_vs_dqn = combined_mean(
    cfr_results,
    "cfr_vs_dqn_seat_0",
    "cfr_vs_dqn_seat_1",
)

cfr_vs_nfsp = combined_mean(
    cfr_results,
    "cfr_vs_nfsp_seat_0",
    "cfr_vs_nfsp_seat_1",
)

agents = ["Random", "DQN", "NFSP", "CFR"]

matrix = [
    [0.0, 0.0, 0.0, 0.0]
    for _ in agents
]


def set_matchup(row_agent, column_agent, payoff):
    row = agents.index(row_agent)
    column = agents.index(column_agent)

    matrix[row][column] = payoff
    matrix[column][row] = -payoff


set_matchup("DQN", "Random", dqn_vs_random)
set_matchup("NFSP", "Random", nfsp_vs_random)
set_matchup("CFR", "Random", cfr_vs_random)
set_matchup("DQN", "NFSP", dqn_vs_nfsp)
set_matchup("CFR", "DQN", cfr_vs_dqn)
set_matchup("CFR", "NFSP", cfr_vs_nfsp)

largest_value = max(
    abs(value)
    for row in matrix
    for value in row
)

color_map = LinearSegmentedColormap.from_list(
    "orange_white_blue",
    ["#D97706", "#FFFFFF", "#2457A7"],
)

figure, axis = plt.subplots(figsize=(8.5, 7))

image = axis.imshow(
    matrix,
    cmap=color_map,
    vmin=-largest_value,
    vmax=largest_value,
)

axis.set_xticks(range(len(agents)))
axis.set_yticks(range(len(agents)))
axis.set_xticklabels(agents)
axis.set_yticklabels(agents)

axis.set_xlabel("Opponent")
axis.set_ylabel("Evaluated agent")
axis.set_title(
    "Head-to-head performance in Leduc Hold'em\n"
    "Average payoff across both seats"
)

for row_index, row in enumerate(matrix):
    for column_index, value in enumerate(row):
        text_color = (
            "white"
            if abs(value) > largest_value * 0.55
            else "#111827"
        )

        axis.text(
            column_index,
            row_index,
            f"{value:+.2f}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=11,
            fontweight="bold",
        )

color_bar = figure.colorbar(image, ax=axis)
color_bar.set_label("Row agent payoff (chips per game)")

figure.text(
    0.01,
    0.01,
    "Positive values favor the row agent. Values average five "
    "evaluation seeds and both player seats using fixed checkpoints.",
    fontsize=8,
    color="#4B5563",
)

figure.tight_layout(rect=(0, 0.06, 1, 1))
figure.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")

print(f"Saved matchup matrix to {OUTPUT_PATH}")