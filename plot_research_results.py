"""Generate publication-ready figures for the poker RL study."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from poker_rl.evaluation import confidence_interval_95

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

DPI = 220

COLORS = {
    "latest": "#6B7280",
    "uniform": "#2563EB",
    "pfsp": "#D97706",
    "uncertainty": "#7C3AED",
    "beta-0": "#2563EB",
    "beta-1": "#7C3AED",
    "beta-2": "#D97706",
    "pool-5": "#2563EB",
    "pool-10": "#7C3AED",
    "pool-20": "#D97706",
}

DISPLAY_NAMES = {
    "latest": "Latest only",
    "uniform": "Uniform",
    "pfsp": "PFSP",
    "uncertainty": "Uncertainty",
    "beta-0": "Beta 0",
    "beta-1": "Beta 1",
    "beta-2": "Beta 2",
    "pool-5": "Pool 5",
    "pool-10": "Pool 10",
    "pool-20": "Pool 20",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "axes.edgecolor": "#374151",
        "axes.labelcolor": "#111827",
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "text.color": "#111827",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as input_file:
        return list(csv.DictReader(input_file))


def save_figure(
    figure: plt.Figure,
    output_name: str,
) -> None:
    output_path = RESULTS_DIR / output_name
    figure.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    print(f"Saved figure to {output_path}")

def plot_primary_aggregate() -> None:
    rows = read_rows(
        RESULTS_DIR / "pool_benchmark_aggregate.csv"
    )
    strategies = [
        "latest",
        "uniform",
        "pfsp",
        "uncertainty",
    ]
    metrics = [
        (
            "reference_mean_payoff",
            "Mean payoff vs reference agents",
        ),
        (
            "reference_worst_case_payoff",
            "Worst-case payoff vs reference agents",
        ),
    ]

    values_by_metric = {
        metric: {
            row["strategy"]: (
                float(row["mean_payoff"]),
                float(row["confidence_interval_95"]),
            )
            for row in rows
            if row["metric"] == metric
        }
        for metric, _ in metrics
    }

    all_intervals = [
        (
            value - interval,
            value + interval,
        )
        for metric, _ in metrics
        for value, interval in (
            values_by_metric[metric].values()
        )
    ]
    minimum = min(
        lower for lower, _ in all_intervals
    )
    maximum = max(
        upper for _, upper in all_intervals
    )
    span = maximum - minimum
    left_limit = minimum - span * 0.08
    right_limit = maximum + span * 0.32

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5.4),
        sharey=True,
    )
    positions = np.arange(len(strategies))

    for axis, (metric, title) in zip(
        axes,
        metrics,
    ):
        for position, strategy in enumerate(
            strategies
        ):
            value, interval = (
                values_by_metric[metric][strategy]
            )

            axis.errorbar(
                value,
                position,
                xerr=interval,
                fmt="o",
                color=COLORS[strategy],
                markeredgecolor="#111827",
                markeredgewidth=0.7,
                markersize=8,
                capsize=4,
                linewidth=1.8,
            )
            axis.text(
                value + interval + span * 0.025,
                position,
                f"{value:+.3f} ± {interval:.3f}",
                va="center",
                fontsize=8.5,
            )

        axis.axvline(
            0,
            color="#374151",
            linewidth=1,
        )
        axis.grid(
            axis="x",
            color="#D1D5DB",
            linewidth=0.7,
            alpha=0.75,
        )
        axis.set_axisbelow(True)
        axis.set_xlim(left_limit, right_limit)
        axis.set_title(title)
        axis.set_xlabel(
            "Average payoff (chips per game)"
        )
        axis.set_yticks(positions)
        axis.set_yticklabels(
            [
                DISPLAY_NAMES[strategy]
                for strategy in strategies
            ]
        )
        axis.invert_yaxis()

    figure.suptitle(
        "Matched-budget opponent-pool performance",
        fontweight="bold",
    )
    figure.text(
        0.01,
        0.015,
        "Points show means across five independent training "
        "seeds; error bars are 95% t-intervals. The fixed "
        "reference population is random, DQN, NFSP, and CFR.",
        fontsize=8,
        color="#4B5563",
    )
    figure.tight_layout(
        rect=(0, 0.07, 1, 0.94)
    )
    save_figure(
        figure,
        "pool_strategy_aggregate.png",
    )
def plot_primary_head_to_head() -> None:
    rows = read_rows(
        RESULTS_DIR / "pool_benchmark_summary.csv"
    )
    strategies = [
        "latest",
        "uniform",
        "pfsp",
        "uncertainty",
    ]
    size = len(strategies)

    matrix = np.zeros((size, size))
    intervals = np.zeros((size, size))

    for row in rows:
        if row["matchup_group"] != "head_to_head":
            continue

        agent_a = row["agent_a"]
        agent_b = row["agent_b"]
        row_index = strategies.index(agent_a)
        column_index = strategies.index(agent_b)
        payoff = float(row["mean_payoff"])
        interval = float(
            row["confidence_interval_95"]
        )

        matrix[row_index, column_index] = payoff
        matrix[column_index, row_index] = -payoff
        intervals[row_index, column_index] = interval
        intervals[column_index, row_index] = interval

    largest_value = float(
        np.max(np.abs(matrix))
    )
    color_map = LinearSegmentedColormap.from_list(
        "orange_white_blue",
        ["#D97706", "#FFFFFF", "#2563EB"],
    )

    figure, axis = plt.subplots(
        figsize=(8.5, 7.3)
    )
    image = axis.imshow(
        matrix,
        cmap=color_map,
        vmin=-largest_value,
        vmax=largest_value,
    )

    labels = [
        DISPLAY_NAMES[strategy]
        for strategy in strategies
    ]
    axis.set_xticks(range(size))
    axis.set_yticks(range(size))
    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels)
    axis.set_xlabel("Opponent")
    axis.set_ylabel("Evaluated strategy")
    axis.set_title(
        "Opponent-pool head-to-head payoff",
        fontweight="bold",
    )

    for row_index in range(size):
        for column_index in range(size):
            if row_index == column_index:
                label = "—"
            else:
                label = (
                    f"{matrix[row_index, column_index]:+.2f}"
                    "\n"
                    f"±{intervals[row_index, column_index]:.2f}"
                )

            text_color = (
                "white"
                if abs(
                    matrix[row_index, column_index]
                )
                > largest_value * 0.58
                else "#111827"
            )
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )

    color_bar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )
    color_bar.set_label(
        "Row-strategy payoff (chips per game)"
    )

    figure.text(
        0.01,
        0.015,
        "Values are mean payoff ± 95% t-interval across "
        "five training seeds, averaged across evaluation seeds "
        "and both seats. Every off-diagonal interval includes zero.",
        fontsize=8,
        color="#4B5563",
    )
    figure.tight_layout(
        rect=(0, 0.07, 1, 1)
    )
    save_figure(
        figure,
        "pool_head_to_head_heatmap.png",
    )
def plot_ablation_aggregate() -> None:
    ablation_rows = read_rows(
        RESULTS_DIR / "ablation_benchmark_aggregate.csv"
    )
    primary_rows = read_rows(
        RESULTS_DIR / "pool_benchmark_aggregate.csv"
    )

    values = {
        (
            row["ablation_label"],
            row["metric"],
        ): (
            float(row["mean_payoff"]),
            float(row["confidence_interval_95"]),
        )
        for row in ablation_rows
    }

    for row in primary_rows:
        if row["strategy"] != "uncertainty":
            continue

        values[
            ("beta-1", row["metric"])
        ] = (
            float(row["mean_payoff"]),
            float(row["confidence_interval_95"]),
        )

    families = [
        (
            "Uncertainty strength",
            ["beta-0", "beta-1", "beta-2"],
        ),
        (
            "Maximum pool size",
            ["pool-5", "pool-10", "pool-20"],
        ),
    ]
    metrics = [
        (
            "reference_mean_payoff",
            "Mean reference payoff",
        ),
        (
            "reference_worst_case_payoff",
            "Worst-case reference payoff",
        ),
    ]

    all_intervals = [
        (
            values[(configuration, metric)][0]
            - values[(configuration, metric)][1],
            values[(configuration, metric)][0]
            + values[(configuration, metric)][1],
        )
        for _, configurations in families
        for configuration in configurations
        for metric, _ in metrics
    ]
    minimum = min(
        lower for lower, _ in all_intervals
    )
    maximum = max(
        upper for _, upper in all_intervals
    )
    span = maximum - minimum
    left_limit = minimum - span * 0.08
    right_limit = maximum + span * 0.32

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8.6),
        sharex=True,
    )

    for row_index, (
        family_name,
        configurations,
    ) in enumerate(families):
        positions = np.arange(
            len(configurations)
        )

        for column_index, (
            metric,
            metric_name,
        ) in enumerate(metrics):
            axis = axes[
                row_index,
                column_index,
            ]

            for position, configuration in (
                enumerate(configurations)
            ):
                value, interval = values[
                    (configuration, metric)
                ]

                axis.errorbar(
                    value,
                    position,
                    xerr=interval,
                    fmt="o",
                    color=COLORS[configuration],
                    markeredgecolor="#111827",
                    markeredgewidth=0.7,
                    markersize=8,
                    capsize=4,
                    linewidth=1.8,
                )
                axis.text(
                    (
                        value
                        + interval
                        + span * 0.025
                    ),
                    position,
                    (
                        f"{value:+.3f} "
                        f"± {interval:.3f}"
                    ),
                    va="center",
                    fontsize=8.5,
                )

            axis.axvline(
                0,
                color="#374151",
                linewidth=1,
            )
            axis.grid(
                axis="x",
                color="#D1D5DB",
                linewidth=0.7,
                alpha=0.75,
            )
            axis.set_axisbelow(True)
            axis.set_xlim(
                left_limit,
                right_limit,
            )
            axis.set_yticks(positions)
            axis.set_yticklabels(
                [
                    DISPLAY_NAMES[configuration]
                    for configuration in (
                        configurations
                    )
                ]
            )
            axis.invert_yaxis()
            axis.set_title(
                f"{family_name}: {metric_name}"
            )
            axis.set_xlabel(
                "Average payoff (chips per game)"
            )

    figure.suptitle(
        "Controlled opponent-pool ablations",
        fontweight="bold",
    )
    figure.text(
        0.01,
        0.012,
        "Points show means across five training seeds; error "
        "bars are 95% t-intervals. Beta 1 is the primary "
        "30-second-checkpoint control. Pool-size runs use "
        "10-second checkpoints.",
        fontsize=8,
        color="#4B5563",
    )
    figure.tight_layout(
        rect=(0, 0.07, 1, 0.95)
    )
    save_figure(
        figure,
        "ablation_aggregate.png",
    )

def plot_training_learning_curves() -> None:
    manifest_rows = read_rows(
        RESULTS_DIR / "pool_training_manifest.csv"
    )
    strategies = [
        "latest",
        "uniform",
        "pfsp",
        "uncertainty",
    ]
    line_styles = {
        "latest": "-",
        "uniform": "--",
        "pfsp": "-.",
        "uncertainty": ":",
    }
    time_grid = np.linspace(
        5,
        300,
        60,
    )
    curves = defaultdict(list)

    for manifest_row in manifest_rows:
        strategy = manifest_row["strategy"]
        run_directory = Path(
            manifest_row["run_directory"]
        )

        if not run_directory.is_absolute():
            run_directory = (
                PROJECT_DIR / run_directory
            )

        progress_rows = read_rows(
            run_directory / "training_progress.csv"
        )
        elapsed_seconds = np.array(
            [
                float(row["elapsed_seconds"])
                for row in progress_rows
            ]
        )
        learner_payoffs = np.array(
            [
                float(row["window_learner_payoff"])
                for row in progress_rows
            ]
        )

        order = np.argsort(elapsed_seconds)
        interpolated = np.interp(
            time_grid,
            elapsed_seconds[order],
            learner_payoffs[order],
        )
        curves[strategy].append(interpolated)

    figure, axis = plt.subplots(
        figsize=(10.5, 6.2)
    )

    for strategy in strategies:
        seed_curves = np.vstack(
            curves[strategy]
        )
        mean_curve = np.mean(
            seed_curves,
            axis=0,
        )
        intervals = np.array(
            [
                confidence_interval_95(
                    seed_curves[
                        :,
                        time_index,
                    ].tolist()
                )
                or 0
                for time_index in range(
                    len(time_grid)
                )
            ]
        )

        axis.plot(
            time_grid,
            mean_curve,
            color=COLORS[strategy],
            linestyle=line_styles[strategy],
            linewidth=2.2,
            label=DISPLAY_NAMES[strategy],
        )
        axis.fill_between(
            time_grid,
            mean_curve - intervals,
            mean_curve + intervals,
            color=COLORS[strategy],
            alpha=0.12,
            linewidth=0,
        )

    axis.axhline(
        0,
        color="#374151",
        linewidth=1,
    )
    axis.grid(
        color="#D1D5DB",
        linewidth=0.7,
        alpha=0.7,
    )
    axis.set_axisbelow(True)
    axis.set_xlim(5, 300)
    axis.set_xlabel(
        "Wall-clock training time (seconds)"
    )
    axis.set_ylabel(
        "Window learner payoff (chips per game)"
    )
    axis.set_title(
        "Opponent-pool training diagnostics",
        fontweight="bold",
    )
    axis.legend(
        loc="upper left",
        ncol=2,
        frameon=False,
    )

    figure.text(
        0.01,
        0.012,
        "Lines show means across five training seeds; shaded "
        "regions are 95% t-intervals. Opponent distributions "
        "change during training and differ by strategy, so these "
        "curves are diagnostics rather than final-policy comparisons.",
        fontsize=8,
        color="#4B5563",
    )
    figure.tight_layout(
        rect=(0, 0.075, 1, 1)
    )
    save_figure(
        figure,
        "training_learning_curves.png",
    )

def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    plot_primary_aggregate()
    plot_primary_head_to_head()
    plot_ablation_aggregate()
    plot_training_learning_curves()


if __name__ == "__main__":
    main()