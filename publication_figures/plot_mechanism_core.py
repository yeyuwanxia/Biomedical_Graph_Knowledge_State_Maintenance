#!/usr/bin/env python3
"""Redraw four manuscript panels from preserved final-figure table inputs.

Run: python publication_figures/plot_mechanism_core.py
Requires: matplotlib, numpy. No experiment is rerun and no interval is estimated.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import tempfile

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kbs_mechanism_core_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

SOURCE_FINAL_PNG_SHA256 = "a0e63428183dd6d2ea01511ec01f3c6defdda4359b52016be272d0f6cc008f74"
INPUT_SHA256 = {
    "table09_mechanism.csv": "3c44efcbf05493cd64c75c66de3dd4e5ac9b4e20e40f6a86ad43df12508462a3",
    "table10_tradeoff.csv": "6f0961667eaa68311cb2f935441a3fd77a58fcd796bd8f54e3b78c04c9d45406",
}
BLUE, TEAL, ROSE, PURPLE = "#256091", "#0E5A4A", "#AD5D56", "#8067B1"


def rows(name):
    file = DATA / name
    if not file.is_file():
        raise FileNotFoundError(f"Missing plotting input: {file}. See publication_figures/README.md.")
    assert hashlib.sha256(file.read_bytes()).hexdigest() == INPUT_SHA256[name], (
        f"Published plotting-input fingerprint differs: {name}. "
        "Use the preserved final-figure report-table input; do not infer values from an image."
    )
    with file.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def style_axis(ax, grid_axis):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.65)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color="#dce2e6", linewidth=0.45)
    ax.tick_params(width=0.65, length=3, labelsize=9)


def trend(ax, data, key, color, title, ylabel, limits):
    x = np.arange(4)
    mean = np.array([float(row[key + "_mean"]) for row in data])
    sd = np.array([float(row[key + "_sd"]) for row in data])
    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.10, linewidth=0)
    ax.errorbar(x, mean, yerr=sd, fmt="o-", color=color, markersize=4.2,
                linewidth=1.35, elinewidth=0.85, capsize=2.5)
    for i, value in enumerate(mean):
        ax.annotate(f"{value:.3f}", (i, value + sd[i]), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, color=color)
    ax.set(xticks=x, xticklabels=["0 (FT)", "0.1", "1", "10"],
           xlabel="Reference multiplier $a$", ylabel=ylabel,
           xlim=(-0.40, 3.40), ylim=limits)
    ax.set_title(title, loc="left", fontsize=10.5, pad=17)
    ax.text(1, 1.035, "Fixed; mean $\\pm$ 1 SD", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.1, color="#4c5860")
    style_axis(ax, "y")
    return {"mean": mean.tolist(), "sd": sd.tolist()}


def main():
    mechanism, tradeoff = rows("table09_mechanism.csv"), rows("table10_tradeoff.csv")
    assert [float(row["a"]) for row in mechanism] == [0, 0.1, 1, 10]
    fixed = [row for row in tradeoff if row["protocol"] == "Fixed"]
    assert [row["outcome"] for row in fixed] == [
        "Protection benefit", "Correction cost", "Net correct change"]
    assert abs(float(fixed[0]["mean"]) - float(fixed[1]["mean"]) - float(fixed[2]["mean"])) < 1e-9
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next(name for name in ["Times New Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"] if name in available)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [font], "font.size": 9.5,
        "axes.labelsize": 9.5, "mathtext.fontset": "stix",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7))
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.115, top=0.90,
                        wspace=0.67, hspace=0.76)
    traces = {}
    traces["a"] = trend(axes[0, 0], mechanism, "retained_l2", BLUE,
                        "(a) Representation drift", "Mean retained-node\nL2 distance", (0, 6.8))
    traces["b"] = trend(axes[0, 1], mechanism, "corrected", TEAL,
                        "(b) Error correction", "Corrected original errors\n(mean count)", (48, 121))
    traces["c"] = trend(axes[1, 0], mechanism, "lost", ROSE,
                        "(c) Lost correct predictions", "Originally correct predictions\nlost (mean count)", (33, 51))
    ax = axes[1, 1]
    for y, (row, color) in enumerate(zip(fixed, [TEAL, ROSE, PURPLE])):
        mean, low, high = [float(row[key]) for key in ["mean", "ci_low", "ci_high"]]
        assert low <= mean <= high
        ax.errorbar(mean, y, xerr=[[mean-low], [high-mean]], fmt="o", color=color,
                    markersize=4.7, elinewidth=1.25, capsize=3)
        ax.annotate(f"{mean:+.3f}", (mean, y), xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, color=color,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
    ax.axvline(0, color="#444444", lw=0.75, ls=(0, (3, 3)))
    ax.set(yticks=[0, 1, 2], yticklabels=["Protection\nbenefit", "Correction\ncost", "Net correct\nchange"],
           ylim=(2.55, -0.65), xlim=(-35, 40), xticks=[-30, -15, 0, 15, 30],
           xlabel="Effect (mean prediction count)")
    ax.set_title("(d) Fixed ($a=1$): Full-old - FT", loc="left", fontsize=10.5, pad=17)
    ax.text(1, 1.035, "Paired 95% $t$ CIs ($df=4$)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.1, color="#4c5860")
    style_axis(ax, "x")
    traces["d"] = [{key: row[key] for key in ["outcome", "mean", "ci_low", "ci_high"]} for row in fixed]
    output = ROOT / "fig1_mechanism_core.png"
    fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    manifest = {
        "source_final_png_sha256": SOURCE_FINAL_PNG_SHA256,
        "input_sha256": INPUT_SHA256,
        "panel_mapping": {"a": "original (a)", "b": "original (b)", "c": "original (c)", "d": "original (f)"},
        "statistics": {"a-c": "mean +/- 1 sample SD across five split means", "d": "paired 95% t confidence intervals; df=4"},
        "source_type": "preserved report-table inputs used to generate the exact final PNG; not per-object predictions",
        "font": font, "dpi": 600, "figure_inches_before_tight_crop": [7.2, 4.7],
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "numeric_trace": traces,
    }
    (ROOT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
