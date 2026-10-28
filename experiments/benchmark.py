"""
MAVIS Benchmark Module
======================
Aggregates and formats benchmark comparisons across all 4 core configurations:
1. M1: Baseline CNN (Raw Catastrophic Forgetting)
2. M2: CNN + EWC (Loss-level parameter regularization)
3. M3: CNN + Attention Memory (Representation-level memory)
4. M4: Full MAVIS (Memory + Attention + EWC)
"""

import os
import sys
import glob
import json
from typing import Dict, Any

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_result(results_dir: str, filename: str) -> Dict[str, Any]:
    candidates = [
        os.path.join(results_dir, "raw", filename),
        os.path.join(results_dir, "raw", "published", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r") as f:
                return json.load(f)
    matches = glob.glob(os.path.join(results_dir, "raw", "**", filename), recursive=True)
    if matches:
        with open(matches[0], "r") as f:
            return json.load(f)
    return {}


def generate_benchmark_table(results_dir: str) -> str:
    """Generates an academic Markdown table comparing available model results."""
    from evaluation.metrics import compute_average_accuracy, compute_forgetting

    baseline = load_result(results_dir, "baseline_results.json")
    ewc = load_result(results_dir, "ewc_results.json")
    mavis = load_result(results_dir, "mavis_results.json")
    mavis_m3 = load_result(results_dir, "mavis_m3_results.json")

    rows = []

    if baseline and "final_accuracy_matrix" in baseline:
        b_acc = compute_average_accuracy(baseline["final_accuracy_matrix"]) * 100
        b_forg = baseline.get("final_forgetting", compute_forgetting(baseline["final_accuracy_matrix"])) * 100
        rows.append(f"| M1: Baseline CNN | None | None | {b_acc:.2f}% | {b_forg:.2f}% |")

    if ewc and "final_accuracy_matrix" in ewc:
        e_acc = compute_average_accuracy(ewc["final_accuracy_matrix"]) * 100
        e_forg = ewc.get("final_forgetting", compute_forgetting(ewc["final_accuracy_matrix"])) * 100
        rows.append(f"| M2: CNN + EWC | EWC (lambda={ewc.get('ewc_lambda', 5000)}) | None | {e_acc:.2f}% | {e_forg:.2f}% |")

    if mavis_m3 and "final_accuracy_matrix_hier" in mavis_m3:
        m3_hier_acc = compute_average_accuracy(mavis_m3["final_accuracy_matrix_hier"]) * 100
        m3_hier_forg = mavis_m3.get("final_forgetting_hier", compute_forgetting(mavis_m3["final_accuracy_matrix_hier"])) * 100
        rows.append(f"| M3: MAVIS (Memory Only, No EWC) | None | Memory Bank (K=10) | {m3_hier_acc:.2f}% | {m3_hier_forg:.2f}% |")

    if mavis and "final_accuracy_matrix_hier" in mavis:
        m_hier_acc = compute_average_accuracy(mavis["final_accuracy_matrix_hier"]) * 100
        m_hier_forg = mavis.get("final_forgetting_hier", compute_forgetting(mavis["final_accuracy_matrix_hier"])) * 100
        rows.append(f"| M4: Full MAVIS (Hierarchical) | EWC (lambda={mavis.get('ewc_lambda', 5000)}) | Memory Bank (K=10) | {m_hier_acc:.2f}% | {m_hier_forg:.2f}% |")

    if mavis and "final_accuracy_matrix_ens" in mavis:
        m_ens_acc = compute_average_accuracy(mavis["final_accuracy_matrix_ens"]) * 100
        m_ens_forg = mavis.get("final_forgetting_ens", compute_forgetting(mavis["final_accuracy_matrix_ens"])) * 100
        rows.append(f"| M4: Full MAVIS (Ensemble) | EWC (lambda={mavis.get('ewc_lambda', 5000)}) | Memory Bank (K=10) | {m_ens_acc:.2f}% | {m_ens_forg:.2f}% |")

    table = [
        "| Architecture | Regularization | External Memory | Final Avg Acc | Average Forgetting |",
        "| :--- | :--- | :--- | :---: | :---: |",
    ] + rows

    return "\n".join(table)


if __name__ == "__main__":
    from config import RESULTS_DIR
    print(generate_benchmark_table(RESULTS_DIR))
