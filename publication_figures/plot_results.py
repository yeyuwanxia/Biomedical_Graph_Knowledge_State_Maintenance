#!/usr/bin/env python3
"""Reproduce three publication figures from the explicitly transcribed PDF tables."""
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kbs_results_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BLUE = "#356B9A"
TEAL = "#208574"
ORANGE = "#C57535"
PURPLE = "#82639D"
GRAY = "#7A858F"
INK = "#25333E"
PALE = "#EFF4F8"
COST_UPDATE = "#8FAAC1"
COST_PREPARATION = "#E0E7ED"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10, "axes.titleweight": "semibold",
    "axes.titlepad": 10, "axes.edgecolor": "#A6AFB7", "axes.linewidth": 0.6,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "grid.color": "#DEE4E9", "grid.linewidth": 0.55,
    "hatch.linewidth": 0.45,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def read(name):
    return pd.read_csv(ROOT / "data" / name)


def clean(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=grid)


def save(fig, name):
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(ROOT / f"{name}.{suffix}", dpi=600 if suffix == "png" else None,
                    bbox_inches="tight", pad_inches=0.12)
    fig.savefig(ROOT / f"{name}_preview.png", dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def validate_data():
    """Check identities and units; do not infer new intervals from rounded SDs."""
    t9, t10 = read("table09_mechanism.csv"), read("table10_tradeoff.csv")
    classes, cost = read("tableA1_classwise.csv"), read("table13_cost.csv")
    assert t9.a.tolist() == [0, 0.1, 1, 10]
    assert len(classes) == 11 and classes.support.sum() == 9407
    assert classes.loc[classes.holm_p < 0.05, "class"].tolist() == ["Anatomy"]
    for _, group in t10.groupby("protocol"):
        means = group.set_index("outcome")["mean"]
        assert np.isclose(means["Protection benefit"] - means["Correction cost"],
                          means["Net correct change"])
    for data, mean_col in [(t10, "mean"), (classes, "difference")]:
        assert (data.ci_low <= data[mean_col]).all()
        assert (data.ci_high >= data[mean_col]).all()
    # Independently rounded component means may differ from a rounded total by 0.01 s.
    assert np.allclose(cost.preparation_mean + cost.cached_update_mean,
                       cost.first_use_mean, atol=0.011, rtol=0)
    assert (cost.service_mean >= cost.first_use_mean).all()
    # Table 11's last row sums the other seven class F1 differences.
    top = classes[classes["class"].isin([
        "Anatomy", "Symptom", "Pathway", "Pharmacologic Class"])]
    assert top.support.sum() == 602
    assert np.isclose(classes.loc[~classes.index.isin(top.index), "difference"].sum(), -1.822)
    print(f"Verified: 11 classes, n={classes.support.sum():,}; "
          f"top-four share={top.difference.sum()/classes.difference.sum():.2%}; "
          f"mean class difference={classes.difference.mean():.4f} pp.")


def trend(ax, data, key, color, title, ylabel, digits, ylim):
    x = np.arange(len(data))
    mean, sd = data[f"{key}_mean"].to_numpy(), data[f"{key}_sd"].to_numpy()
    ax.fill_between(x, mean-sd, mean+sd, color=color, alpha=0.13, linewidth=0)
    ax.errorbar(x, mean, yerr=sd, color=color, marker="o", markersize=4.3,
                linewidth=1.6, elinewidth=0.7, capsize=2.4)
    for i, value in enumerate(mean):
        # Place labels above the upper SD cap, rather than on the error-bar stem.
        horizontal_offset = 2.5 if i == 0 else -2.5 if i == len(mean)-1 else 0
        # The last two lost-correct means have similar upper SD endpoints;
        # stagger the final label to keep both legible in the narrower top panels.
        vertical_offset = 17 if key == "lost" and i == len(mean)-1 else 5
        ax.annotate(f"{value:.{digits}f}", (i, value + sd[i]), xytext=(horizontal_offset, vertical_offset),
                    textcoords="offset points", ha="center",
                    va="bottom", fontsize=9,
                    color=color, fontweight="medium")
    ax.set(xticks=x, xticklabels=["0 (FT)", "0.1", "1", "10"],
           xlabel="Reference multiplier $a$", ylabel=ylabel, ylim=ylim, xlim=(-0.28, 3.28))
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    ax.tick_params(labelsize=9)
    clean(ax)


def mechanism():
    data, tradeoff = read("table09_mechanism.csv"), read("table10_tradeoff.csv")
    fig = plt.figure(figsize=(7.8, 8.8))
    grid = fig.add_gridspec(3, 1, left=0.09, right=0.98, bottom=0.145, top=0.87,
                           hspace=0.72, height_ratios=[1, 1, 1.15])
    # Give the three long y-axis labels their own horizontal space, independently
    # of the two-column rows, and reserve a separate band above the footer.
    top = grid[0].subgridspec(1, 3, wspace=0.60)
    middle = grid[1].subgridspec(1, 2, wspace=0.30)
    bottom = grid[2].subgridspec(1, 2, wspace=0.30)
    fig.text(0.09, 0.957, "Representation stability and prediction trade-offs",
             fontsize=13, weight="semibold")
    fig.text(0.09, 0.924, "Hetionet node classification  |  Fixed strength sweep and paired Full-old–FT contrasts",
             fontsize=8.8, color=GRAY)
    specs = [
        (top[0, 0], "retained_l2", BLUE, "(a) Representation\ndrift", "Mean retained-node\nL2 distance", 3, (0, 6.5)),
        (top[0, 1], "corrected", TEAL, "(b) Error correction", "Corrected original\nerrors (mean count)", 3, (48, 118)),
        (top[0, 2], "lost", ORANGE, "(c) Lost correct\npredictions", "Originally correct\npredictions lost\n(mean count)", 3, (33, 50)),
        (middle[0, 0], "macro_f1", BLUE, "(d) Task quality", "Macro-F1 (%)", 2, (49.5, 58.5)),
        (middle[0, 1], "js_x1000", PURPLE, "(e) Retrain proximity", "JS to Matched-Retrain\n" + r"($10^{-3}$ nats)", 2, (1.8, 4.9)),
    ]
    for position, key, color, title, ylabel, digits, ylim in specs:
        trend(fig.add_subplot(position), data, key, color, title, ylabel, digits, ylim)
    colors = [TEAL, ORANGE, PURPLE]
    for i, protocol in enumerate(["Fixed", "Tuned"]):
        ax = fig.add_subplot(bottom[0, i])
        part = tradeoff[tradeoff.protocol == protocol].reset_index(drop=True)
        for y, row in part.iterrows():
            ax.errorbar(row["mean"], y,
                        xerr=[[row["mean"]-row.ci_low], [row.ci_high-row["mean"]]],
                        fmt="o", color=colors[y], capsize=3, markersize=5, elinewidth=1.5)
            ax.annotate(f"{row['mean']:+.3f}", (row["mean"], y),
                        xytext=(0, 9), textcoords="offset points", ha="center",
                        fontsize=9, color=colors[y], zorder=4,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2}
                        if protocol == "Fixed" and y == 0 else None)
        ax.axvline(0, color=INK, lw=0.9, ls=(0, (3, 3)))
        ax.set(yticks=[0, 1, 2], yticklabels=["Protection\nbenefit", "Correction\ncost", "Net correct\nchange"],
               ylim=(2.6, -0.65), xlim=(-45, 62), xticks=[-40, -20, 0, 20, 40, 60],
               xlabel="Signed effect (mean prediction count)")
        title = r"(f) Fixed ($a=1$)" if protocol == "Fixed" else "(g) Tuned"
        ax.set_title(title, loc="left", fontsize=11)
        ax.xaxis.label.set_size(10)
        ax.tick_params(labelsize=9)
        clean(ax, "x")
    fig.text(0.09, 0.0735, "(a–e) Mean ± 1 SD across five split means; multiplier values are equally spaced categories.",
             fontsize=8.8, color=GRAY)
    fig.text(0.09, 0.0525, "(f–g) Paired 95% t intervals (df = 4).",
             fontsize=8.8, color=GRAY)
    save(fig, "fig1_mechanism_tradeoff")


def classwise():
    data = read("tableA1_classwise.csv").sort_values("difference").reset_index(drop=True)
    fig = plt.figure(figsize=(7.8, 5.2))
    grid = fig.add_gridspec(1, 3, width_ratios=[2.9, 3.4, 1.55],
                           left=0.035, right=0.985, bottom=0.23, top=0.77, wspace=0.065)
    labels, ax, numbers = [fig.add_subplot(grid[0, i]) for i in range(3)]
    fig.text(0.035, 0.94, "Classwise F1 cost of preservation", fontsize=13, weight="semibold")
    fig.text(0.035, 0.895, "Tuned protocol  |  Full-old − FT  |  All eleven Hetionet entity classes",
             fontsize=9, color=GRAY)
    for panel in (labels, ax, numbers):
        panel.set_ylim(10.65, -0.65)
        for y in range(4):
            panel.axhspan(y-0.48, y+0.48, color=PALE, zorder=0)
    labels.set(xlim=(0, 1))
    labels.axis("off")
    numbers.set(xlim=(0, 1))
    numbers.axis("off")
    labels.text(0, 1.045, "Entity class", transform=labels.transAxes, weight="semibold", fontsize=8.8)
    labels.text(0.99, 1.045, "True-label support", transform=labels.transAxes, ha="right", weight="semibold", fontsize=8.8)
    ax.text(0.5, 1.045, "F1 difference (pp)", transform=ax.transAxes, ha="center", weight="semibold", fontsize=8.8)
    numbers.text(0.22, 1.045, "Mean", transform=numbers.transAxes, ha="center", weight="semibold", fontsize=8.8)
    numbers.text(0.81, 1.045, "Holm p", transform=numbers.transAxes, ha="center", weight="semibold", fontsize=8.8)
    for y, row in data.iterrows():
        color = BLUE if y < 4 else GRAY
        label = row["class"] + (" *" if row.holm_p < 0.05 else "")
        labels.text(0, y, label, va="center", fontsize=8.4,
                    weight="semibold" if row.holm_p < 0.05 else "normal")
        labels.text(0.99, y, f"{row.support:,}", ha="right", va="center", fontsize=8.1, color=GRAY)
        ax.errorbar(row.difference, y,
                    xerr=[[row.difference-row.ci_low], [row.ci_high-row.difference]],
                    fmt="o", markersize=5 if y < 4 else 4,
                    color=color, capsize=2.6, elinewidth=1.3, zorder=3)
        numbers.text(0.22, y, f"{row.difference:+.3f}", va="center", ha="center", fontsize=8.3, color=color)
        numbers.text(0.81, y, f"{row.holm_p:.4f}", va="center", ha="center", fontsize=8.0,
                     weight="semibold" if row.holm_p < 0.05 else "normal")
    ax.axvline(0, color=INK, lw=0.9, ls=(0, (3, 3)))
    ax.set(xlim=(-4.3, 2.4), xticks=[-4, -3, -2, -1, 0, 1, 2], yticks=[],
           xlabel="Class F1 difference (Full-old − FT, pp)")
    clean(ax, "x")
    ax.spines["left"].set_visible(False)
    fig.text(0.035, 0.115, "Intervals: unadjusted paired 95% t CIs across five split means (df = 4).", fontsize=8.1, color=GRAY)
    fig.text(0.035, 0.0825, "* Holm-adjusted p < 0.05 across eleven class-F1 tests.", fontsize=8.1, color=GRAY)
    save(fig, "fig2_classwise_cost")


def cost():
    data = read("table13_cost.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.9), sharey=True,
                             gridspec_kw={"wspace": 0.16})
    fig.subplots_adjust(left=0.20, right=0.99, bottom=0.22, top=0.73)
    fig.text(0.035, 0.94, "Preparation-inclusive maintenance cost", fontsize=13, weight="semibold")
    fig.text(0.035, 0.892, "Tuned protocol  |  First use = cached update + preparation  |  Hetionet  |  Time (s)",
             fontsize=9, color=GRAY)
    fig.legend(handles=[Patch(facecolor=COST_UPDATE, label="Cached update"),
                        Patch(facecolor=COST_PREPARATION, hatch="//", edgecolor=GRAY, linewidth=0.35, label="Preparation"),
                        Line2D([0], [0], color=INK, linewidth=1, marker="|", label="First-use ± 1 SD")],
               loc="upper left", bbox_to_anchor=(0.026, 0.861), frameon=False, ncol=3,
               fontsize=8.5, handlelength=2.0, columnspacing=1.9)
    for i, (ax, task) in enumerate(zip(axes, ["Node", "Relation"])):
        part = data[data.task == task].reset_index(drop=True)
        y = np.arange(len(part))
        ax.axhspan(2.55, 3.45, color=PALE, zorder=0)
        ax.barh(y, part.cached_update_mean, color=COST_UPDATE, height=0.61, zorder=2)
        ax.barh(y, part.preparation_mean, left=part.cached_update_mean,
                color=COST_PREPARATION, edgecolor=GRAY, linewidth=0.35, hatch="//", height=0.61, zorder=2)
        ax.errorbar(part.first_use_mean, y, xerr=part.first_use_sd,
                    fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.5, zorder=3)
        for row_y, row in part.iterrows():
            ax.annotate(f"{row.first_use_mean:.2f}", (row.first_use_mean + row.first_use_sd, row_y),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                        weight="semibold" if row.method == "Full-old" else "normal")
            if row.preparation_mean > 1:
                ax.text(row.cached_update_mean + row.preparation_mean/2, row_y,
                        f"{row.preparation_mean:.2f}", ha="center", va="center",
                        fontsize=7.6, color=INK, weight="semibold",
                        bbox={"facecolor": COST_PREPARATION, "edgecolor": "none", "pad": 0.6})
                ax.text(row.cached_update_mean/2, row_y, f"{row.cached_update_mean:.2f}",
                        ha="center", va="center", fontsize=7.6, color=INK)
        ax.set(xlim=(0, 52), xticks=[0, 10, 20, 30, 40, 50], yticks=y,
               yticklabels=part.method, ylim=(6.65, -0.65), xlabel="First-use time (s)")
        if i == 0:
            ax.set_ylabel("Method", labelpad=10)
        ax.set_title(f"({'ab'[i]}) {task}", loc="left")
        ax.tick_params(axis="y", length=0, pad=7, labelsize=9)
        clean(ax, "x")
        ax.spines["left"].set_visible(False)
    fig.text(0.035, 0.10, "Whiskers use the reported first-use SD across five split means, including component covariance.", fontsize=8.0, color=GRAY)
    save(fig, "fig3_maintenance_cost")


def export_cost_appendix():
    """Retain all four timing scopes and their reported SDs for the appendix."""
    data = read("table13_cost.csv")
    lines = [
        "% Complete original Table 13, including Service. No additional packages required.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Maintenance cost under the Tuned protocol, in seconds "
        r"(mean $\pm$ sample SD across five split means). First use includes "
        r"preparation and cached-update work; service additionally includes model "
        r"loading or creation and final inference. Development search and "
        r"research-only multi-checkpoint diagnostic overhead are outside these scopes.}",
        r"\label{tab:maintenance-cost-full}",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"Task & Method & Preparation & Cached update & First use & Service \\",
        r"\hline",
    ]
    for index, row in data.iterrows():
        if index == 7:
            lines.append(r"\hline")
        timings = [f"${row[f'{scope}_mean']:.2f} \\pm {row[f'{scope}_sd']:.2f}$"
                   for scope in ["preparation", "cached_update", "first_use", "service"]]
        lines.append(" & ".join([row.task, row.method, *timings]) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    (ROOT / "appendix_table13.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    validate_data()
    mechanism()
    classwise()
    cost()
    export_cost_appendix()
    print(f"Saved PNG (600 dpi), vector PDF/SVG and previews to {ROOT}")
