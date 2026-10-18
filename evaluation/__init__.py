"""MAVIS Evaluation Package — Accuracy, forgetting, metrics, confusion matrix, plots."""

from evaluation.metrics import (
    compute_average_accuracy,
    compute_forgetting,
    compute_backward_transfer,
    compute_learning_accuracy,
    summarize_metrics,
)
from evaluation.visualize import visualize_task_samples, plot_accuracy_curves

__all__ = [
    "compute_average_accuracy",
    "compute_forgetting",
    "compute_backward_transfer",
    "compute_learning_accuracy",
    "summarize_metrics",
    "visualize_task_samples",
    "plot_accuracy_curves",
]
