"""
MAVIS Evaluation Metrics Module
===============================
Standard continual learning evaluation metrics:
1. Average Accuracy (ACC): Mean accuracy across seen tasks after training task T.
2. Backward Transfer / Forgetting (BWT / F): Performance drop on older tasks.
3. Learning Accuracy (LA): Immediate performance on task t right after learning it.
"""

from typing import List, Dict, Any
import numpy as np


def compute_average_accuracy(accuracy_matrix: List[List[float]]) -> float:
    """
    Computes average accuracy across all tasks evaluated at the final step.
    ACC = (1 / T) * sum_{j=1}^T R_{T, j}
    """
    if not accuracy_matrix:
        return 0.0
    final_row = accuracy_matrix[-1]
    return float(np.mean(final_row))


def compute_forgetting(accuracy_matrix: List[List[float]]) -> float:
    """
    Computes average catastrophic forgetting across past tasks.
    F = (1 / (T - 1)) * sum_{j=1}^{T-1} (max_{t in {j..T-1}} R_{t, j} - R_{T, j})
    """
    if not accuracy_matrix or len(accuracy_matrix) < 2:
        return 0.0

    num_tasks = len(accuracy_matrix)
    forgetting = 0.0
    for j in range(num_tasks - 1):
        max_past_acc = max(accuracy_matrix[i][j] for i in range(j, num_tasks - 1))
        current_acc = accuracy_matrix[-1][j]
        forgetting += max_past_acc - current_acc

    return float(forgetting / (num_tasks - 1))


def compute_backward_transfer(accuracy_matrix: List[List[float]]) -> float:
    """
    Computes Backward Transfer (BWT).
    BWT = (1 / (T - 1)) * sum_{j=1}^{T-1} (R_{T, j} - R_{j, j})
    Positive BWT indicates backward improvement; negative BWT indicates forgetting.
    """
    if not accuracy_matrix or len(accuracy_matrix) < 2:
        return 0.0

    num_tasks = len(accuracy_matrix)
    bwt = 0.0
    for j in range(num_tasks - 1):
        bwt += accuracy_matrix[-1][j] - accuracy_matrix[j][j]

    return float(bwt / (num_tasks - 1))


def compute_learning_accuracy(accuracy_matrix: List[List[float]]) -> float:
    """
    Computes Learning Accuracy (LA) — average initial task accuracy.
    LA = (1 / T) * sum_{j=1}^T R_{j, j}
    """
    if not accuracy_matrix:
        return 0.0
    return float(np.mean([accuracy_matrix[i][i] for i in range(len(accuracy_matrix))]))


def summarize_metrics(accuracy_matrix: List[List[float]]) -> Dict[str, float]:
    """Returns a dictionary summarizing all core continual learning metrics."""
    return {
        "average_accuracy": compute_average_accuracy(accuracy_matrix),
        "average_forgetting": compute_forgetting(accuracy_matrix),
        "backward_transfer": compute_backward_transfer(accuracy_matrix),
        "learning_accuracy": compute_learning_accuracy(accuracy_matrix),
    }
