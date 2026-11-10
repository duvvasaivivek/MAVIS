"""
Figures for the factorial study
===============================
Reads the same tagged result files as experiments/analyze_factorial.py and writes figures
plus a CSV beside each one, so every plotted number is also available as text.

This script does not import anything the training path imports, and the training path never
imports it. It is safe to run or edit while a training job is in flight.

Everything it plots comes out of the saved result JSON, so there is nothing to instrument
beforehand. It works on a partial run: cells with no result file yet are simply left out, so
it is worth running once the first seed finishes rather than waiting for all three.

Run:
    python experiments/plot_factorial.py --width 128 --tasks 10 --epochs 50
    python experiments/plot_factorial.py --width 32 --tasks 5 --epochs 10     # the quick sweep
    python experiments/plot_factorial.py --width 128 --tasks 10 --epochs 50 --outdir results/figures

Colour: the five cells use the first five slots of a categorical palette validated for
colour-vision deficiency (worst adjacent CVD delta-E 9.1, normal-vision 19.6). Three of those
slots fall below 3:1 contrast on a white surface, so every series also carries a distinct
marker shape and a legend, and each figure is accompanied by its CSV.
"""

import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")           # no display on the cluster
import matplotlib.pyplot as plt

from experiments.analyze_factorial import PREDICTORS, load_cells, final_acc, forgetting

# --- palette -----------------------------------------------------------------------------
# Categorical slots 1-5, fixed order, never cycled. One colour per cell for the whole figure
# set, so a cell keeps its identity from plot to plot.
CELL_COLOR = {
    "M1":    "#2a78d6",   # blue
    "M2":    "#eb6834",   # orange
    "M3":    "#1baf7a",   # aqua
    "M4":    "#eda100",   # yellow
    "M4noA": "#e87ba4",   # magenta
}
# Secondary encoding, because three of those colours are low-contrast on white.
CELL_MARKER = {"M1": "o", "M2": "s", "M3": "^", "M4": "D", "M4noA": "v"}

CELL_LABEL = {
    "M1": "M1  baseline",
    "M2": "M2  + EWC",
    "M3": "M3  memory only",
    "M4": "M4  full MAVIS",
    "M4noA": "M4  no retrieval",
}

# Sequential ramp for the accuracy matrices: one hue, light to dark. Never a rainbow.
SEQUENTIAL = "Blues"

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dedcd6"

PLOT_ORDER = ["M1", "M2", "M3", "M4", "M4noA"]


def style(ax):
    """Recessive grid and axes; text in ink, never in a series colour."""
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.xaxis.label.set_color(INK_SOFT)
    ax.yaxis.label.set_color(INK_SOFT)
    ax.title.set_color(INK)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main_predictor(cell):
    """The predictor each cell is summarised by: the plain head for M1/M2, hierarchical for MAVIS."""
    return "head" if cell in ("M1", "M2") else "hier"


def curve(matrix):
    """Average accuracy over the tasks seen so far, after each task."""
    return [float(np.mean(row)) for row in matrix]


# --- figure 1: retention over the task sequence -------------------------------------------
def plot_curves(cells, outdir, scale_note):
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=160)
    style(ax)
    rows, any_data = [], False

    for cell in PLOT_ORDER:
        runs = cells.get(cell) or {}
        key = PREDICTORS[cell][main_predictor(cell)]
        series = [curve(d[key]) for d in runs.values() if key in d]
        if not series:
            continue
        any_data = True
        n_tasks = min(len(s) for s in series)
        arr = np.array([s[:n_tasks] for s in series]) * 100
        mean, x = arr.mean(axis=0), np.arange(1, n_tasks + 1)
        ax.plot(x, mean, color=CELL_COLOR[cell], linewidth=2.0,
                marker=CELL_MARKER[cell], markersize=5.5,
                markeredgecolor="white", markeredgewidth=1.2,
                label=f"{CELL_LABEL[cell]}  (n={len(series)})")
        if len(series) > 1:                     # seed spread as a band, not error bars
            ax.fill_between(x, arr.min(axis=0), arr.max(axis=0),
                            color=CELL_COLOR[cell], alpha=0.15, linewidth=0)
        for i, v in enumerate(mean):
            rows.append([cell, i + 1, round(float(v), 2), len(series)])

    if not any_data:
        plt.close(fig)
        return None

    ax.set_xlabel("tasks learned so far")
    ax.set_ylabel("average accuracy over all seen classes (%)")
    ax.set_title("Retention across the task sequence", fontsize=12, pad=12)
    ax.set_ylim(bottom=0)
    # Task counts are whole numbers; the default locator produces 1.5, 2.5 and so on.
    n_max = int(max(line.get_xdata().max() for line in ax.lines))
    ax.set_xticks(np.arange(1, n_max + 1))
    ax.set_xlim(0.8, n_max + 0.2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc="upper right")
    fig.text(0.01, 0.01, scale_note, fontsize=7.5, color=INK_SOFT)
    fig.tight_layout()

    png = os.path.join(outdir, "retention_curves.png")
    fig.savefig(png, facecolor="white")
    plt.close(fig)
    write_csv(os.path.join(outdir, "retention_curves.csv"),
              ["cell", "tasks_seen", "mean_accuracy_pct", "seeds"], rows)
    return png


# --- figure 2: final accuracy and forgetting ----------------------------------------------
def plot_summary(cells, outdir, scale_note):
    """Two panels, never two y-axes on one plot: accuracy and forgetting are different measures."""
    present = [c for c in PLOT_ORDER if (cells.get(c) or {})]
    if not present:
        return None

    acc, forg, labels, colors, rows = [], [], [], [], []
    acc_err, forg_err = [], []
    for cell in present:
        runs = cells[cell]
        key = PREDICTORS[cell][main_predictor(cell)]
        a = [100 * final_acc(d[key]) for d in runs.values() if key in d]
        f = [100 * forgetting(d[key]) for d in runs.values() if key in d]
        if not a:
            continue
        acc.append(np.mean(a)); forg.append(np.mean(f))
        acc_err.append(np.ptp(a) / 2 if len(a) > 1 else 0.0)
        forg_err.append(np.ptp(f) / 2 if len(f) > 1 else 0.0)
        labels.append(CELL_LABEL[cell]); colors.append(CELL_COLOR[cell])
        rows.append([cell, round(np.mean(a), 2), round(np.std(a, ddof=1), 2) if len(a) > 1 else "",
                     round(np.mean(f), 2), len(a)])

    # Height follows the number of cells so the bars keep a sensible thickness whether the
    # run is partway through with two cells or complete with five.
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 1.5 + 0.62 * len(labels)), dpi=160)
    y = np.arange(len(labels))
    for ax, vals, errs, title, xlabel in (
        (axes[0], acc, acc_err, "Final accuracy (higher is better)", "accuracy after the last task (%)"),
        (axes[1], forg, forg_err, "Forgetting (lower is better)", "average forgetting (%)"),
    ):
        style(ax)
        ax.barh(y, vals, height=0.62, color=colors, edgecolor="white", linewidth=2.0)
        if any(e > 0 for e in errs):
            ax.errorbar(vals, y, xerr=errs, fmt="none", ecolor=INK_SOFT, elinewidth=1.2, capsize=3)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9, color=INK)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=11, pad=10)
        ax.grid(axis="y", visible=False)
        for yi, v in zip(y, vals):          # direct labels: the contrast relief rule
            ax.text(v + max(vals) * 0.015, yi, f"{v:.2f}", va="center",
                    fontsize=9, color=INK)
        ax.set_xlim(0, max(vals) * 1.18)

    fig.text(0.01, 0.01, scale_note + "   error bars show the seed range where more than one seed exists",
             fontsize=7.5, color=INK_SOFT)
    fig.tight_layout()
    png = os.path.join(outdir, "final_summary.png")
    fig.savefig(png, facecolor="white")
    plt.close(fig)
    write_csv(os.path.join(outdir, "final_summary.csv"),
              ["cell", "final_accuracy_pct", "seed_sd", "forgetting_pct", "seeds"], rows)
    return png


# --- figure 3: one accuracy matrix per cell ------------------------------------------------
def plot_matrices(cells, outdir, scale_note):
    present = [c for c in PLOT_ORDER if (cells.get(c) or {})]
    if not present:
        return None

    fig, axes = plt.subplots(1, len(present), figsize=(3.1 * len(present), 3.5), dpi=160,
                             squeeze=False)
    rows = []
    for ax, cell in zip(axes[0], present):
        runs = cells[cell]
        seed = sorted(runs)[0]
        key = PREDICTORS[cell][main_predictor(cell)]
        m = runs[seed][key]
        T = len(m)
        grid = np.full((T, T), np.nan)
        for i, row in enumerate(m):
            for j, v in enumerate(row):
                grid[i, j] = 100 * v
                rows.append([cell, seed, i + 1, j + 1, round(100 * v, 2)])
        # Tasks not yet learned get their own flat grey. Without this they render the same
        # near-white as a genuine zero, and "never seen" would read as "forgotten completely".
        cmap = plt.get_cmap(SEQUENTIAL).copy()
        cmap.set_bad("#e8e7e2")
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=np.nanmax(grid), aspect="equal")
        ax.set_title(f"{CELL_LABEL[cell]}\nseed {seed}", fontsize=9.5, color=INK, pad=8)
        ax.set_xlabel("accuracy on task", fontsize=8.5, color=INK_SOFT)
        ax.set_ylabel("after training task", fontsize=8.5, color=INK_SOFT)
        ticks = np.arange(T)
        ax.set_xticks(ticks); ax.set_xticklabels(ticks + 1, fontsize=7, color=INK_SOFT)
        ax.set_yticks(ticks); ax.set_yticklabels(ticks + 1, fontsize=7, color=INK_SOFT)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)

    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("accuracy (%)", fontsize=8.5, color=INK_SOFT)
    cbar.ax.tick_params(colors=INK_SOFT, labelsize=7.5)
    cbar.outline.set_visible(False)
    fig.suptitle("Accuracy on every past task, after every task", fontsize=12, color=INK)
    fig.text(0.01, 0.01, scale_note + "   the bottom row of each grid is the final state",
             fontsize=7.5, color=INK_SOFT)

    png = os.path.join(outdir, "accuracy_matrices.png")
    fig.savefig(png, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    write_csv(os.path.join(outdir, "accuracy_matrices.csv"),
              ["cell", "seed", "after_task", "on_task", "accuracy_pct"], rows)
    return png


# --- figure 4: the four predictors, MAVIS cells only --------------------------------------
def plot_predictors(cells, outdir, scale_note):
    mavis = [c for c in ("M3", "M4", "M4noA") if (cells.get(c) or {})]
    if not mavis:
        return None
    preds = ["head", "ncm", "ens", "hier"]
    pretty = {"head": "linear head", "ncm": "nearest prototype",
              "ens": "ensemble", "hier": "hierarchical"}

    fig, ax = plt.subplots(figsize=(7.8, 4.2), dpi=160)
    style(ax)
    width = 0.8 / len(mavis)
    x = np.arange(len(preds))
    rows = []
    top = 0.0
    for k, cell in enumerate(mavis):
        runs = cells[cell]
        vals, errs = [], []
        for p in preds:
            key = PREDICTORS[cell].get(p)
            a = [100 * final_acc(d[key]) for d in runs.values() if key and key in d]
            vals.append(np.mean(a) if a else 0.0)
            errs.append(np.ptp(a) / 2 if len(a) > 1 else 0.0)
            rows.append([cell, p, round(float(np.mean(a)), 2) if a else "", len(a)])
        pos = x - 0.4 + width * (k + 0.5)
        ax.bar(pos, vals, width=width * 0.9, color=CELL_COLOR[cell],
               edgecolor="white", linewidth=2.0, label=CELL_LABEL[cell])
        if any(e > 0 for e in errs):
            ax.errorbar(pos, vals, yerr=errs, fmt="none", ecolor=INK_SOFT,
                        elinewidth=1.2, capsize=3)
        # Clear the error-bar cap, not just the bar top, or the number sits on the whisker.
        for p, v, e in zip(pos, vals, errs):
            ax.text(p, v + e + max(vals) * 0.035, f"{v:.1f}", ha="center",
                    fontsize=8, color=INK)
        top = max(top, max(v + e for v, e in zip(vals, errs)))

    ax.set_xticks(x); ax.set_xticklabels([pretty[p] for p in preds], fontsize=9.5, color=INK)
    ax.set_ylabel("final accuracy (%)")
    ax.set_title("How the four predictors compare, within each MAVIS variant", fontsize=11.5, pad=12)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, top * 1.22)          # headroom so the legend never sits on a bar
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc="upper right")
    fig.text(0.01, 0.01, scale_note, fontsize=7.5, color=INK_SOFT)
    fig.tight_layout()
    png = os.path.join(outdir, "predictors.png")
    fig.savefig(png, facecolor="white")
    plt.close(fig)
    write_csv(os.path.join(outdir, "predictors.csv"),
              ["cell", "predictor", "final_accuracy_pct", "seeds"], rows)
    return png


# --- figure 5: is the memory path actually contributing? -----------------------------------
def plot_retrieval_share(cells, outdir, scale_note):
    """The diagnostic the project never had: how much of the fused feature comes from memory."""
    # Aggregate over seeds, one line per cell. Nine labelled lines would swamp the legend.
    series = {}
    for cell in ("M3", "M4", "M4noA"):
        runs = [d.get("norm_diagnostics") for d in (cells.get(cell) or {}).values()
                if d.get("norm_diagnostics")]
        if runs:
            series[cell] = runs
    if not series:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=160)
    rows = []
    for ax, field, title, ylabel in (
        (axes[0], "r_share", "Share of the fused feature that comes from memory",
         "retrieved branch share of the squared length"),
        (axes[1], "gate_mean", "What the fusion gate is doing",
         "mean gate value (1.0 = live feature only)"),
    ):
        style(ax)
        for cell in ("M3", "M4", "M4noA"):
            runs = series.get(cell)
            if not runs:
                continue
            n = min(len(r) for r in runs)
            arr = np.array([[e[field] for e in r[:n]] for r in runs])
            xs = np.array([runs[0][i].get("after_task", i + 1) for i in range(n)])
            mean = arr.mean(axis=0)
            ax.plot(xs, mean, color=CELL_COLOR[cell], linewidth=2.0,
                    marker=CELL_MARKER[cell], markersize=5.5,
                    markeredgecolor="white", markeredgewidth=1.2,
                    label=f"{CELL_LABEL[cell]}  (n={len(runs)})")
            if len(runs) > 1:
                ax.fill_between(xs, arr.min(axis=0), arr.max(axis=0),
                                color=CELL_COLOR[cell], alpha=0.15, linewidth=0)
            if field == "r_share":
                for xi, yi in zip(xs, mean):
                    rows.append([cell, len(runs), int(xi), round(float(yi), 4)])
        ax.set_xlabel("after training task")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_ylim(0, 1.0)
        n_max = int(max(line.get_xdata().max() for line in ax.lines))
        ax.set_xticks(np.arange(1, n_max + 1))
        ax.set_xlim(0.8, n_max + 0.2)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_SOFT)
    fig.text(0.01, 0.01, scale_note + "   before the 2026-10-25 fixes this fell to about 0.02",
             fontsize=7.5, color=INK_SOFT)
    fig.tight_layout()
    png = os.path.join(outdir, "retrieval_share.png")
    fig.savefig(png, facecolor="white")
    plt.close(fig)
    write_csv(os.path.join(outdir, "retrieval_share.csv"),
              ["cell", "seeds", "after_task", "mean_retrieved_share"], rows)
    return png


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--tag-extra", type=str, default="")
    ap.add_argument("--outdir", type=str, default=os.path.join("results", "figures"))
    args = ap.parse_args()

    tag = f"_fac{args.tag_extra}_w{args.width}_t{args.tasks}_e{args.epochs}_s*"
    cells = load_cells(tag)

    found = {c: len(r) for c, r in cells.items() if r}
    if not found:
        print(f"No result files matching {tag}. Nothing to plot yet.")
        return
    print(f"tag {tag}")
    print("runs found: " + ", ".join(f"{c}={n}" for c, n in found.items()))
    missing = [c for c in PLOT_ORDER if c not in found]
    if missing:
        print(f"still missing: {', '.join(missing)} - the figures below leave them out")

    os.makedirs(args.outdir, exist_ok=True)
    scale_note = (f"Tiny ImageNet, {args.tasks} tasks of 20 classes, "
                  f"encoder width {args.width}, {args.epochs} epochs per task")

    print()
    for fn in (plot_curves, plot_summary, plot_matrices, plot_predictors, plot_retrieval_share):
        png = fn(cells, args.outdir, scale_note)
        if png:
            print(f"  wrote {png}  (+ .csv)")
        else:
            print(f"  {fn.__name__}: not enough data yet, skipped")


if __name__ == "__main__":
    main()
