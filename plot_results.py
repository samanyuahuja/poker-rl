import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "results" / "baseline_evaluation.csv"
OUTPUT_PATH = PROJECT_DIR / "results" / "baseline_summary.png"

scores = defaultdict(list)

with CSV_PATH.open() as csv_file:
    reader = csv.DictReader(csv_file)

    for row in reader:
        scores[row["matchup"]].append(float(row["average_payoff"]))

# The sign is reversed for nfsp_vs_dqn so the chart reports
# the DQN's payoff while it occupies seat 1.
chart_rows = [
    ("dqn_vs_random_seat_0", "DQN vs random (seat 0)", 1, "#2457A7"),
    ("dqn_vs_random_seat_1", "DQN vs random (seat 1)", 1, "#2457A7"),
    ("nfsp_vs_random_seat_0", "NFSP vs random (seat 0)", 1, "#D97706"),
    ("nfsp_vs_random_seat_1", "NFSP vs random (seat 1)", 1, "#D97706"),
    ("dqn_vs_nfsp", "DQN vs NFSP (seat 0)", 1, "#2457A7"),
    ("nfsp_vs_dqn", "DQN vs NFSP (seat 1)", -1, "#2457A7"),
]

labels = []
means = []
intervals = []
colors = []

for matchup, label, sign, color in chart_rows:
    values = [sign * value for value in scores[matchup]]

    mean = statistics.mean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    confidence_interval = 2.776 * standard_error

    labels.append(label)
    means.append(mean)
    intervals.append(confidence_interval)
    colors.append(color)

figure, axis = plt.subplots(figsize=(10, 6.5))

positions = range(len(labels))

bars = axis.barh(
    positions,
    means,
    xerr=intervals,
    color=colors,
    edgecolor="#1F2937",
    linewidth=0.8,
    capsize=4,
)

axis.set_yticks(list(positions))
axis.set_yticklabels(labels)
axis.invert_yaxis()

axis.set_xlabel("Average payoff (chips per game)")
axis.set_title(
    "Fixed-checkpoint performance in Leduc Hold'em\n"
    "Mean over 5 evaluation seeds, 2,000 games per seed"
)

axis.axvline(0, color="#374151", linewidth=1)
axis.grid(axis="x", color="#D1D5DB", linewidth=0.7, alpha=0.7)
axis.set_axisbelow(True)

largest_value = max(
    mean + interval
    for mean, interval in zip(means, intervals)
)
axis.set_xlim(0, largest_value * 1.28)

for bar, mean, interval in zip(bars, means, intervals):
    axis.text(
        mean + interval + 0.025,
        bar.get_y() + bar.get_height() / 2,
        f"{mean:.2f} ± {interval:.2f}",
        va="center",
        fontsize=9,
        color="#111827",
    )

figure.text(
    0.01,
    0.01,
    "Error bars are 95% t-intervals across evaluation seeds for fixed "
    "checkpoints; they do not include retraining variance.",
    fontsize=8,
    color="#4B5563",
)

figure.tight_layout(rect=(0, 0.06, 1, 1))
figure.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")

print(f"Saved chart to {OUTPUT_PATH}")