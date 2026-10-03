"""
Elastic Weight Consolidation (EWC)
===================================
Computes the Fisher Information Matrix and applies a quadratic penalty
to prevent important weights from changing when learning new tasks.

Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in
neural networks", PNAS 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_fisher(model, dataloader, device, num_samples=1000):
    """
    Estimate the diagonal Fisher Information Matrix.

    For each parameter, Fisher_i = E[ (d log p(y|x) / d theta_i)^2 ]
    which simplifies to the mean of squared gradients of the log-likelihood
    over the training data.

    Args:
        model: trained model (kept in eval mode for stable BN stats)
        dataloader: training data for the task just completed
        device: 'cuda' or 'cpu'
        num_samples: max number of samples to use for estimation

    Returns:
        Dictionary mapping parameter name -> Fisher diagonal tensor
    """
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

    samples_seen = 0
    for inputs, targets in dataloader:
        if samples_seen >= num_samples:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)

        model.zero_grad()
        outputs = model(inputs)
        # Use the log-softmax for numerical stability
        log_probs = F.log_softmax(outputs, dim=1)
        # Select the log-prob of the true class
        loss = F.nll_loss(log_probs, targets)
        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += (p.grad.detach() ** 2) * batch_size

        samples_seen += batch_size

    # Average over number of samples
    for n in fisher:
        fisher[n] /= samples_seen

    model.train()
    return fisher


def snapshot_params(model):
    """
    Save a copy of all current model parameters.

    Returns:
        Dictionary mapping parameter name -> detached tensor clone
    """
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def compute_ewc_loss(model, fisher_list, params_list, ewc_lambda):
    """
    Compute the EWC penalty across all previously completed tasks.

    L_ewc = (lambda / 2) * SUM_tasks SUM_params F_i * (theta_i - theta_i*)^2

    Args:
        model: current model being trained
        fisher_list: list of Fisher dicts, one per completed task
        params_list: list of param snapshot dicts, one per completed task
        ewc_lambda: regularization strength

    Returns:
        Scalar EWC loss tensor
    """
    ewc_loss = torch.tensor(0.0, device=next(model.parameters()).device)

    for fisher, old_params in zip(fisher_list, params_list):
        for n, p in model.named_parameters():
            if p.requires_grad and n in fisher:
                ewc_loss += (fisher[n] * (p - old_params[n]) ** 2).sum()

    return (ewc_lambda / 2.0) * ewc_loss
