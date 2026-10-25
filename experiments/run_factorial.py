"""
Review factorial driver (added 2026-10-25)
==========================================
Runs the five cells of the review experiment with the SAME architecture, schedule, class order
and seeds, so the cells are actually comparable (the logged M1/M2 files were not):

    M1       train_baseline
    M2       train_ewc            (gamma = config default, lambda = --ewc-lambda)
    M3       train_mavis --no-ewc
    M4       train_mavis
    M4-noA   train_mavis --no-attention   (forward-pass memory path removed; everything else kept)

Every run is tagged  _fac_w{width}_t{tasks}_e{epochs}_s{seed}  so nothing overwrites the
published result files. Runs that already have a results file are skipped (resume-friendly).

Example (what was used for the review):
    python experiments/run_factorial.py --width 64 --tasks 5 --epochs 6 --seeds 1 2 3 --batch-size 128
"""

import os
import sys
import time
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

CELLS = {
    "M1": ("train_baseline", []),
    "M2": ("train_ewc", []),
    "M3": ("train_mavis", ["--no-ewc", "--fresh"]),
    "M4": ("train_mavis", ["--fresh"]),
    "M4noA": ("train_mavis", ["--no-attention", "--fresh"]),
}
RESULT_PREFIX = {"M1": "baseline", "M2": "ewc", "M3": "mavis_m3", "M4": "mavis", "M4noA": "mavis_noattn"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--cells", nargs="+", default=list(CELLS), choices=list(CELLS))
    ap.add_argument("--ewc-lambda-m2", type=float, default=None, help="lambda for M2 (default: config default)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log_dir = os.path.join(ROOT, "results", "factorial_logs")
    os.makedirs(log_dir, exist_ok=True)

    for seed in args.seeds:
        for cell in args.cells:
            cmd_name, extra = CELLS[cell]
            tag = f"_fac_w{args.width}_t{args.tasks}_e{args.epochs}_s{seed}"
            res_file = os.path.join(ROOT, "results", "raw", f"{RESULT_PREFIX[cell]}{tag}_results.json")
            if os.path.exists(res_file):
                print(f"[skip] {cell} seed {seed}: {res_file} exists", flush=True)
                continue
            cmd = [PY, os.path.join(ROOT, "main.py"), cmd_name,
                   "--seed", str(seed), "--tasks", str(args.tasks), "--epochs", str(args.epochs),
                   "--width", str(args.width), "--batch-size", str(args.batch_size), "--tag", tag] + extra
            if cell == "M2" and args.ewc_lambda_m2 is not None:
                cmd += ["--ewc-lambda", str(args.ewc_lambda_m2)]
            print(f"\n[run] {cell} seed {seed}: {' '.join(cmd)}", flush=True)
            if args.dry_run:
                continue
            t0 = time.time()
            log_path = os.path.join(log_dir, f"{cell}{tag}.log")
            with open(log_path, "w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            print(f"[done] {cell} seed {seed}: exit {proc.returncode}, {(time.time()-t0)/60:.1f} min, log {log_path}", flush=True)


if __name__ == "__main__":
    main()
