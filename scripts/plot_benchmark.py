"""Render aggregate benchmark evidence from a completed report, without refitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new figure output path")
    report = json.loads(args.report.read_text())
    if report.get("validation_status") != "FINAL_FROZEN_PROTOCOL_FULL_109_PARTICIPANTS":
        raise ValueError("This evidence figure requires the full frozen benchmark")
    selected = report["final_holdout"]["selected"]
    candidates = {item["candidate"]: item for item in report["selection"]["candidate_results"]}
    ordered = sorted(candidates, key=lambda name: candidates[name]["mean_balanced_accuracy"])
    names = [name.replace("_", " ") for name in ordered]
    means = [100 * candidates[name]["mean_balanced_accuracy"] for name in ordered]
    deviations = [100 * candidates[name]["std_balanced_accuracy"] for name in ordered]
    colors = ["#167d7f" if name == selected["candidate"] else "#9ab3c5" for name in ordered]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    left.barh(names, means, xerr=deviations, color=colors, capsize=3)
    left.set_xlim(0, 100)
    left.set_xlabel("Balanced accuracy (%) · mean ± fold standard deviation")
    left.set_title("Model selection · training participants only", loc="left", pad=14)
    left.grid(axis="x", alpha=0.18)
    left.set_axisbelow(True)
    matrix = np.asarray(selected["confusion_matrix"])
    right.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    for row in range(2):
        for column in range(2):
            right.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=22,
                color="white" if matrix[row, column] > matrix.max() / 2 else "#15324c",
            )
    labels = [name.replace("eyes_", "Eyes ") for name in selected["confusion_matrix_labels"]]
    right.set_xticks([0, 1], labels)
    right.set_yticks([0, 1], labels)
    right.set_xlabel("Predicted condition")
    right.set_ylabel("Recorded condition")
    holdout_participants = len(report["split"]["validation_participant_ids"])
    holdout_windows = report["split"]["n_validation_windows"]
    resamples = selected["confidence_intervals"]["replicates"]
    right.set_title(
        f"Final holdout · {holdout_participants} unseen participants", loc="left", pad=14
    )
    score = 100 * selected["balanced_accuracy"]
    interval = selected["confidence_intervals"]["intervals"]["balanced_accuracy"]
    fig.suptitle(
        f"Eyes-open / eyes-closed EEG · {score:.1f}% held-out balanced accuracy",
        fontsize=17,
        fontweight="bold",
        x=0.03,
        ha="left",
    )
    fig.text(
        0.03,
        0.015,
        f"95% participant-cluster CI: {100 * interval['low']:.1f}–{100 * interval['high']:.1f}%"
        f" · {resamples:,} resamples · {holdout_windows:,} held-out windows\n"
        "PhysioNet EEGMMIDB 1.0.0 · same-dataset offline evaluation; fixed run-order confounding remains.",
        fontsize=9,
        color="#405265",
    )
    fig.tight_layout(rect=(0.01, 0.10, 0.99, 0.91), w_pad=4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
