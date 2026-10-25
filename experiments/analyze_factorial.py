"""
Review factorial analysis (added 2026-10-25)
============================================
Reads results/raw/*_fac_*_results.json produced by experiments/run_factorial.py and reports,
for every cell and predictor:

  * final average accuracy per seed, mean, SD, 95% CI across seeds (t distribution)
  * per-image binomial standard error (the within-run noise floor)
  * average forgetting per seed
  * pairwise contrasts (M4-M3, M4-M4noA, M2-M1, M3-M1) with paired-by-seed CI and Cohen's d
  * the factorial interaction (M4-M3) - (M2-M1) with paired-by-seed CI
  * minimum detectable effect (MDE) for the seeds actually run, and seeds needed for 1, 2, 3 points

Nothing here is hard-coded from the docs; every number is read from the result files.

Usage:
    python experiments/analyze_factorial.py --width 64 --tasks 5 --epochs 6
    python experiments/analyze_factorial.py --published     # CIs for the logged M1..M4 files
"""

import os
import sys
import glob
import json
import math
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")

# two-sided 97.5% t quantiles, df = 1..30
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        25: 2.060, 30: 2.042}
Z80 = 0.8416  # one-sided 80% normal quantile (power)


def t975(df):
    if df <= 0:
        return float("nan")
    if df in T975:
        return T975[df]
    keys = sorted(T975)
    for k in keys:
        if k > df:
            return T975[k]
    return 1.96


def final_acc(matrix):
    return float(np.mean(matrix[-1]))


def forgetting(matrix):
    T = len(matrix)
    if T < 2:
        return 0.0
    return float(np.mean([max(matrix[i][j] for i in range(j, T - 1)) - matrix[-1][j] for j in range(T - 1)]))


def binomial_se(matrix, images_per_task=1000):
    """SE of the final average accuracy treating each task's accuracy as Binomial(images_per_task)."""
    row = matrix[-1]
    var = sum(p * (1 - p) / images_per_task for p in row) / (len(row) ** 2)
    return math.sqrt(var)


def ci(values):
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        return (float("nan"),) * 4
    m = v.mean()
    sd = v.std(ddof=1) if n > 1 else float("nan")
    half = t975(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return m, sd, m - half, m + half


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else float("nan")


def mde_two_sample(sd, n, alpha_t=None):
    """Minimum detectable difference (two-sample, equal n, alpha .05 two-sided, power .80)."""
    if n < 2 or not np.isfinite(sd):
        return float("nan")
    t_a = t975(2 * n - 2) if alpha_t is None else alpha_t
    return (t_a + Z80) * sd * math.sqrt(2.0 / n)


def seeds_needed(sd, delta):
    if not np.isfinite(sd) or delta <= 0:
        return float("nan")
    return 2 * ((1.96 + Z80) * sd / delta) ** 2


PREDICTORS = {
    "M1": {"head": "final_accuracy_matrix"},
    "M2": {"head": "final_accuracy_matrix"},
    "M3": {"hier": "final_accuracy_matrix_hier", "ens": "final_accuracy_matrix_ens",
           "head": "final_accuracy_matrix_head", "ncm": "final_accuracy_matrix_ncm"},
    "M4": {"hier": "final_accuracy_matrix_hier", "ens": "final_accuracy_matrix_ens",
           "head": "final_accuracy_matrix_head", "ncm": "final_accuracy_matrix_ncm"},
    "M4noA": {"hier": "final_accuracy_matrix_hier", "ens": "final_accuracy_matrix_ens",
              "head": "final_accuracy_matrix_head", "ncm": "final_accuracy_matrix_ncm"},
}
PREFIX = {"M1": "baseline", "M2": "ewc", "M3": "mavis_m3", "M4": "mavis", "M4noA": "mavis_noattn"}


def load_cells(tag_glob):
    cells = {}
    for cell, prefix in PREFIX.items():
        files = sorted(glob.glob(os.path.join(RAW, f"{prefix}{tag_glob}_results.json")))
        runs = {}
        for f in files:
            d = json.load(open(f))
            runs[d.get("seed", os.path.basename(f))] = d
        cells[cell] = runs
    return cells


def report_published():
    print("=" * 78)
    print("PUBLISHED RESULT FILES (one unseeded run each): final average accuracy with per-image 95% CI")
    print("=" * 78)
    files = {"M1 baseline_results.json": ("baseline_results.json", "final_accuracy_matrix"),
             "M2 ewc_results.json": ("ewc_results.json", "final_accuracy_matrix"),
             "M3 hier": ("mavis_m3_results.json", "final_accuracy_matrix_hier"),
             "M3 ens": ("mavis_m3_results.json", "final_accuracy_matrix_ens"),
             "M3 head": ("mavis_m3_results.json", "final_accuracy_matrix_head"),
             "M3 ncm": ("mavis_m3_results.json", "final_accuracy_matrix_ncm"),
             "M4 hier": ("mavis_results.json", "final_accuracy_matrix_hier"),
             "M4 ens": ("mavis_results.json", "final_accuracy_matrix_ens"),
             "M4 head": ("mavis_results.json", "final_accuracy_matrix_head"),
             "M4 ncm": ("mavis_results.json", "final_accuracy_matrix_ncm")}
    accs = {}
    for label, (fn, key) in files.items():
        p = os.path.join(RAW, fn)
        if not os.path.exists(p):
            continue
        m = json.load(open(p))[key]
        a, se = final_acc(m), binomial_se(m)
        accs[label] = (a, se, m)
        print(f"{label:28s} ACC={100*a:6.2f}%  per-image SE={100*se:.2f}  95% CI=[{100*(a-1.96*se):.2f}, {100*(a+1.96*se):.2f}]  forgetting={100*forgetting(m):.2f}%")
    def contrast(a, b):
        (xa, sa, _), (xb, sb, _) = accs[a], accs[b]
        d = xa - xb; se = math.sqrt(sa ** 2 + sb ** 2)
        return d, se
    print("\nContrasts on per-image noise only (ignores seed-to-seed variance, which is unknown with 1 run per cell):")
    for a, b in [("M4 hier", "M3 hier"), ("M4 head", "M3 head"), ("M4 ncm", "M3 ncm"), ("M2 ewc_results.json", "M1 baseline_results.json"), ("M3 hier", "M1 baseline_results.json")]:
        if a in accs and b in accs:
            d, se = contrast(a, b)
            print(f"  {a} - {b}: {100*d:+.2f} pts, per-image 95% CI [{100*(d-1.96*se):+.2f}, {100*(d+1.96*se):+.2f}]")
    if all(k in accs for k in ["M4 hier", "M3 hier", "M2 ewc_results.json", "M1 baseline_results.json"]):
        d1, s1 = contrast("M4 hier", "M3 hier"); d2, s2 = contrast("M2 ewc_results.json", "M1 baseline_results.json")
        inter = d1 - d2; se = math.sqrt(s1 ** 2 + s2 ** 2)
        print(f"  interaction (M4-M3)-(M2-M1): {100*inter:+.2f} pts, per-image 95% CI [{100*(inter-1.96*se):+.2f}, {100*(inter+1.96*se):+.2f}]")
    print("\nMinimum detectable difference between two single runs on per-image noise alone (n=10,000 images each, p~0.2):")
    se_single = math.sqrt(2 * 0.2 * 0.8 / 10000)
    print(f"  {100*(1.96+Z80)*se_single:.2f} points. Seed-to-seed variance is NOT included because every cell has one run.")
    print("NOTE: M1 and M2 files come from a different architecture and a label-offset bug (see audit); contrasts with them are not meaningful.")


def report_factorial(tag_glob, images_per_task):
    cells = load_cells(tag_glob)
    print("=" * 78)
    print(f"FACTORIAL STUDY  tag={tag_glob}")
    print("=" * 78)
    acc_by = {}   # (cell, predictor) -> {seed: acc}
    forg_by = {}
    for cell, runs in cells.items():
        if not runs:
            print(f"{cell}: no runs found")
            continue
        for pred, key in PREDICTORS[cell].items():
            acc_by[(cell, pred)] = {s: final_acc(d[key]) for s, d in runs.items()}
            forg_by[(cell, pred)] = {s: forgetting(d[key]) for s, d in runs.items()}
            vals = list(acc_by[(cell, pred)].values())
            m, sd, lo, hi = ci(vals)
            se_img = np.mean([binomial_se(d[key], images_per_task) for d in runs.values()])
            # With one seed there is no sample SD, so print it as unavailable rather than NaN.
            spread = (f"SD={100*sd:5.2f}  95% CI=[{100*lo:6.2f}, {100*hi:6.2f}]" if len(vals) > 1
                      else "SD=  n/a  95% CI=  n/a (one seed)")
            print(f"{cell:6s} {pred:5s} n={len(vals)}  ACC mean={100*m:6.2f}%  {spread}  "
                  f"per-image SE={100*se_img:.2f}  forgetting mean={100*np.mean(list(forg_by[(cell, pred)].values())):.2f}%  "
                  f"per-seed={[round(100*v, 2) for v in vals]}")
        lams = sorted({d.get("ewc_lambda") for d in runs.values() if d.get("ewc_lambda") is not None})
        if lams:
            print(f"       ewc_lambda used: {lams}  (published M2 used 50000; config default 5000 is calibrated for M4)")
        if cell in ("M3", "M4", "M4noA"):
            for s, d in runs.items():
                diag = d.get("norm_diagnostics", [])
                if diag:
                    last = diag[-1]
                    print(f"       seed {s} norms after last task: ||f||={last['f_norm']:.2f} ||r||={last['r_norm']:.2f} gate={last['gate_mean']:.3f} retrieved share={last['r_share']:.4f}")

    def paired(a_key, b_key):
        a, b = acc_by.get(a_key, {}), acc_by.get(b_key, {})
        seeds = sorted(set(a) & set(b))
        if not seeds:
            return None
        diffs = [a[s] - b[s] for s in seeds]
        m, sd, lo, hi = ci(diffs)
        return m, sd, lo, hi, len(seeds), cohens_d([a[s] for s in seeds], [b[s] for s in seeds])

    print("\nContrasts (paired by seed; 95% t CI across seeds; Cohen's d on unpaired SD):")
    for name, a_key, b_key in [
        ("EWC on top of memory  (M4 - M3), hier", ("M4", "hier"), ("M3", "hier")),
        ("EWC on top of memory  (M4 - M3), head", ("M4", "head"), ("M3", "head")),
        ("EWC on top of memory  (M4 - M3), ncm", ("M4", "ncm"), ("M3", "ncm")),
        ("Attention path        (M4 - M4noA), hier", ("M4", "hier"), ("M4noA", "hier")),
        ("Attention path        (M4 - M4noA), head", ("M4", "head"), ("M4noA", "head")),
        ("Attention path        (M4 - M4noA), ncm", ("M4", "ncm"), ("M4noA", "ncm")),
        ("EWC alone             (M2 - M1), head", ("M2", "head"), ("M1", "head")),
        ("Memory pipeline alone (M3 - M1), hier vs head", ("M3", "hier"), ("M1", "head")),
        ("Full vs baseline      (M4 - M1), hier vs head", ("M4", "hier"), ("M1", "head")),
    ]:
        r = paired(a_key, b_key)
        if r is None:
            print(f"  {name}: missing runs")
            continue
        m, sd, lo, hi, n, d = r
        if n < 2 or math.isnan(lo) or math.isnan(hi):
            # One seed gives no interval. Printing "null not rejected" here would be a
            # conclusion drawn from missing data, so report the point estimate and refuse it.
            print(f"  {name}: {100*m:+.2f} pts  95% CI n/a  n={n}  d=n/a"
                  f"  -> point estimate only, seed variance unmeasured")
            continue
        verdict = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0 (null not rejected)"
        print(f"  {name}: {100*m:+.2f} pts  95% CI [{100*lo:+.2f}, {100*hi:+.2f}]  n={n}  d={d:.2f}  -> {verdict}")

    # interaction
    a = acc_by.get(("M4", "hier"), {}); b = acc_by.get(("M3", "hier"), {}); c = acc_by.get(("M2", "head"), {}); e = acc_by.get(("M1", "head"), {})
    seeds = sorted(set(a) & set(b) & set(c) & set(e))
    if seeds:
        inter = [(a[s] - b[s]) - (c[s] - e[s]) for s in seeds]
        m, sd, lo, hi = ci(inter)
        if len(seeds) < 2 or math.isnan(lo):
            print()
            print(f"Interaction (M4-M3)-(M2-M1): {100*m:+.2f} pts  95% CI n/a  "
                  f"n={len(seeds)}  -> point estimate only, seed variance unmeasured")
        else:
            print(f"\nInteraction (M4-M3)-(M2-M1): {100*m:+.2f} pts  95% CI [{100*lo:+.2f}, {100*hi:+.2f}]  n={len(seeds)}  "
                  f"-> {'super-additive' if lo > 0 else ('sub-additive' if hi < 0 else 'no evidence of non-additivity')}")

    # MDE from the pooled seed SD of M4 hier and M3 hier
    sds = [np.std(list(v.values()), ddof=1) for k, v in acc_by.items() if len(v) > 1 and k[1] in ("hier", "head")]
    if sds:
        sd_pool = float(np.sqrt(np.mean(np.square(sds))))
        n = max(len(v) for v in acc_by.values())
        print(f"\nPooled seed-to-seed SD of final ACC: {100*sd_pool:.2f} points (over {len(sds)} cell/predictor series)")
        print(f"Minimum detectable difference with n={n} seeds per cell (alpha .05 two-sided, power .80): {100*mde_two_sample(sd_pool, n):.2f} points")
        for delta in (0.01, 0.02, 0.03, 0.05):
            print(f"  seeds per cell needed to detect {100*delta:.0f} point(s): {seeds_needed(sd_pool, delta):.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--images-per-task", type=int, default=1000)
    ap.add_argument("--published", action="store_true", help="analyse the logged M1..M4 files instead")
    args = ap.parse_args()
    if args.published:
        report_published()
    else:
        report_factorial(f"_fac_w{args.width}_t{args.tasks}_e{args.epochs}_s*", args.images_per_task)
