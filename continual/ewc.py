"""
Elastic Weight Consolidation (EWC) -- Online Variant
=====================================================
Maintains a single running Fisher Information Matrix instead of
accumulating separate matrices for every past task. This prevents
the penalty from being diluted across too many competing constraints.

Reference:
  - Kirkpatrick et al., "Overcoming catastrophic forgetting", PNAS 2017
  - Schwarz et al., "Progress & Compress", ICML 2018 (Online EWC)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_fisher(model, dataloader, device, num_samples=1000):
    """
    Estimate the diagonal Fisher Information Matrix.

    For each parameter, Fisher_i = E[ (d log p(y|x) / d theta_i)^2 ]

    Args:
        model: trained model
        dataloader: training data for the task just completed
        device: 'cuda' or 'cpu'
        num_samples: max samples to use for estimation

    Returns:
        Dictionary mapping parameter name -> Fisher diagonal tensor
    """
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

    samples_seen = 0
    batches_seen = 0
    for inputs, targets in dataloader:
        if samples_seen >= num_samples:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)

        model.zero_grad()
        outputs = model(inputs)
        log_probs = F.log_softmax(outputs, dim=1)
        loss = F.nll_loss(log_probs, targets)
        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2

        samples_seen += batch_size
        batches_seen += 1

    # Average over batches
    for n in fisher:
        fisher[n] /= batches_seen

    model.train()
    return fisher


def snapshot_params(model):
    """Save a copy of all current model parameters."""
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


class OnlineEWC:
    """
    Online EWC: maintains a single running Fisher and a single
    reference parameter snapshot, updated after each task.

    Instead of storing N separate (Fisher, params) pairs:
        F_running = gamma * F_running + F_new_task
        theta_star = current params after task
    """

    def __init__(self, model, ewc_lambda=1000.0, gamma=0.95):
        """
        Args:
            model: the model to protect
            ewc_lambda: regularization strength
            gamma: decay factor for older Fisher contributions (0-1).
                   Lower = forget older tasks faster, Higher = remember more.
        """
        self.ewc_lambda = ewc_lambda
        self.gamma = gamma
        self.running_fisher = None
        self.saved_params = None
        self.task_count = 0

    def update(self, model, dataloader, device, num_samples=1000):
        """
        Call this AFTER training on a task completes.
        Updates the running Fisher and saves current params.
        """
        new_fisher = compute_fisher(model, dataloader, device, num_samples)

        if self.running_fisher is None:
            # First task: just store directly
            self.running_fisher = new_fisher
        else:
            # Blend: decay old Fisher, add new
            for n in self.running_fisher:
                self.running_fisher[n] = (
                    self.gamma * self.running_fisher[n] + new_fisher[n]
                )

        # Always snapshot the latest params
        self.saved_params = snapshot_params(model)
        self.task_count += 1

    def penalty(self, model):
        """
        Compute the EWC penalty term.

        Returns:
            Scalar tensor: (lambda/2) * sum(F_i * (theta_i - theta_i*)^2)
        """
        if self.running_fisher is None or self.saved_params is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.running_fisher:
                loss += (self.running_fisher[n] * (p - self.saved_params[n]) ** 2).sum()

        return (self.ewc_lambda / 2.0) * loss
