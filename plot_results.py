"""Create analysis figures and source tables from completed study outputs."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from src.analysis import normal_inference

WINDOW_TITLES = {
    "0_30": "0–29",
    "30_120": "30–119",
    "120_360": "120–359",
    "360_720": "360–719",
}

LABELS = {
    "test_based_control": "Test-based control",
    "unrestricted_control": "Unrestricted control",
    "negative_control_exposure": "Negative control exposure",
    "synthetic_example": "Synthetic example",
}
COLORS = ["#3878a8", "#d18b30", "#a34d75"]


def plot_results(result_paths, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in result_paths:
        frame = pd.read_csv(Path(path) / "risk_estimates.csv")
        frame["study"] = Path(path).name
        frames.append(frame)
    all_results = pd.concat(frames, ignore_index=True)
    all_results.to_csv(output / "all_risk_estimates.csv", index=False)
    window_order = ["30_120", "120_360", "360_720"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n = len(result_paths)
    for i, frame in enumerate(frames):
        counts = (
            frame.loc[~frame["negative"]]
            .groupby("window")["attributable"]
            .sum()
            .reindex(window_order)
        )
        positions = np.arange(3) + (i - (n - 1) / 2) * 0.7 / n
        ax.bar(
            positions,
            counts,
            width=0.7 / n,
            color=COLORS[i % len(COLORS)],
            label=LABELS.get(frame["study"].iloc[0], frame["study"].iloc[0]),
        )
    ax.set_xticks(np.arange(3), ["30–119", "120–359", "360–719"])
    ax.set_xlabel("Days after enrollment")
    ax.set_ylabel("Attributable outcomes")
    ax.set_ylim(bottom=0, top=max(1, ax.get_ylim()[1]))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "attributable_outcome_counts.pdf")
    fig.savefig(output / "attributable_outcome_counts.png", dpi=200)
    plt.close(fig)
    for frame in frames:
        study = frame["study"].iloc[0]
        for window, subset in frame.groupby("window", sort=False):
            # A stable alphabetical order also makes full outcome tables easy to inspect.
            subset = subset.sort_values("OUTCOME")
            for page, start in enumerate(range(0, len(subset), 30), 1):
                rows = subset.iloc[start : start + 30]
                positions = np.arange(len(rows))
                fig, axes = plt.subplots(
                    1, 2, figsize=(14, max(4, len(rows) * 0.23)), sharey=True
                )
                for ax, metric, null in [(axes[0], "RR", 1), (axes[1], "RD", 0)]:
                    valid = rows[f"{metric}_valid"]
                    ax.errorbar(
                        rows.loc[valid, metric],
                        positions[valid],
                        xerr=rows.loc[valid, f"{metric}_CI_UPPER_BONFERRONI"]
                        - rows.loc[valid, metric],
                        fmt="o",
                        markersize=3,
                        capsize=2,
                        color=COLORS[0],
                    )
                    ax.axvline(null, color="0.5", linestyle=":")
                    ax.set_xlabel(
                        "Risk ratio"
                        if metric == "RR"
                        else "Risk difference (per month)"
                    )
                axes[0].set_yticks(positions, rows["OUTCOME"].str.replace("_", " "))
                axes[0].invert_yaxis()
                fig.suptitle(
                    f"{LABELS.get(study, study)} · days {WINDOW_TITLES[window]}\nBonferroni-adjusted 95% intervals"
                )
                fig.tight_layout()
                fig.savefig(
                    output / f"{study}_{window}_outcomes_{page:02d}.pdf",
                    bbox_inches="tight",
                )
                plt.close(fig)
    return all_results


def plot_monthly_trajectories(result_path, outcomes, output):
    result_path, output = Path(result_path), Path(output)
    folder = result_path / "timeseries" / "bootstrap_dataframes"
    files = sorted(
        (p for p in folder.glob("*.csv") if p.stem.isdigit()), key=lambda p: int(p.stem)
    )
    frames = [pd.read_csv(p).set_index("Y_label") for p in files]
    if len(frames) < 3 or files[0].stem != "0":
        raise ValueError(
            "Monthly trajectories require a point estimate and bootstrap replicates."
        )
    fig, ax = plt.subplots(figsize=(10, 5))
    exported = []
    starts = list(range(30, 720, 30))
    for outcome in outcomes:
        labels = [f"{outcome}_Day_{day}_{day + 30}" for day in starts]
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.stack(
                [
                    (f.loc[labels, "Y_trt"] / f.loc[labels, "Y_ctrl"]).to_numpy()
                    for f in frames
                ]
            )
        statistics = normal_inference(values[0], values[1:], 1.0, 1)
        x = np.array(starts)
        ax.plot(x, values[0], label=outcome.replace("_", " "))
        ax.fill_between(
            x, statistics["CI_LOWER_95"], statistics["CI_UPPER_95"], alpha=0.15
        )
        exported.append(
            pd.DataFrame(
                {"OUTCOME": outcome, "day_start": x, "RR": values[0], **statistics}
            )
        )
    ax.axhline(1, color="0.5", linestyle=":")
    ax.set_xlabel("Days after enrollment (start of 30-day bin)")
    ax.set_ylabel("Risk ratio")
    ax.legend(frameon=False)
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "monthly_trajectories.pdf")
    fig.savefig(output / "monthly_trajectories.png", dpi=200)
    pd.concat(exported).to_csv(output / "monthly_trajectories.csv", index=False)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", nargs="+", help="Completed study output directories."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--outcomes",
        nargs="+",
        help="Optional monthly trajectories, using the first results directory.",
    )
    args = parser.parse_args()
    plot_results(args.results, args.output)
    if args.outcomes:
        plot_monthly_trajectories(args.results[0], args.outcomes, args.output)


if __name__ == "__main__":
    main()
