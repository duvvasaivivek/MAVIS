"""
Continual Learning Trainer (PyTorch)
====================================
Handles sequential training of tasks and tracks catastrophic forgetting.
Supports both baseline (no regularization) and EWC modes.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from continual.ewc import compute_fisher, snapshot_params, compute_ewc_loss


class ContinualTrainer:
    """
    Trains a model sequentially across continual learning tasks.

    Args:
        model: PyTorch model (e.g., BaselineCNN)
        loader: TinyImageNetLoader providing per-task DataLoaders
        task_gen: TaskGenerator defining the task splits
        learning_rate: optimizer learning rate
        epochs_per_task: training epochs per task
        checkpoint_dir: directory to save model checkpoints
        device: 'cuda' or 'cpu'
        ewc_lambda: EWC regularization strength (0 = baseline, >0 = EWC)
        fisher_samples: number of samples used to estimate Fisher
    """

    def __init__(self, model, loader, task_gen, learning_rate, epochs_per_task,
                 checkpoint_dir, device=None, ewc_lambda=0.0, fisher_samples=1000):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = model.to(self.device)
        self.loader = loader
        self.task_gen = task_gen
        self.learning_rate = learning_rate
        self.epochs_per_task = epochs_per_task
        self.checkpoint_dir = checkpoint_dir
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_matrix = []

        # EWC state: accumulated Fisher and param snapshots from all past tasks
        self.fisher_list = []
        self.params_list = []

        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_task(self, task_id: int):
        mode = "EWC" if self.ewc_lambda > 0 else "Baseline"
        print(f"\n[{self.device.upper()}] --- Training Task {task_id + 1} ({mode}) ---")
        train_loader = self.loader.get_task_train_dataloader(task_id)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scaler = torch.amp.GradScaler('cuda', enabled=(self.device == "cuda"))

        for epoch in range(self.epochs_per_task):
            self.model.train()
            total_loss = 0
            total_ewc = 0
            correct = 0
            total = 0

            start_time = time.time()

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
                    outputs = self.model(inputs)
                    ce_loss = self.criterion(outputs, targets)

                    # Add EWC penalty if active
                    if self.ewc_lambda > 0 and len(self.fisher_list) > 0:
                        ewc_loss = compute_ewc_loss(
                            self.model, self.fisher_list, self.params_list, self.ewc_lambda
                        )
                        loss = ce_loss + ewc_loss
                        total_ewc += ewc_loss.item() * inputs.size(0)
                    else:
                        loss = ce_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += ce_loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            epoch_loss = total_loss / total
            epoch_acc = correct / total
            time_taken = time.time() - start_time

            if self.ewc_lambda > 0 and len(self.fisher_list) > 0:
                epoch_ewc = total_ewc / total
                print(f"  Epoch {epoch+1:02d}/{self.epochs_per_task:02d} | "
                      f"CE: {epoch_loss:.4f} | EWC: {epoch_ewc:.4f} | "
                      f"Acc: {epoch_acc:.4f} | Time: {time_taken:.1f}s")
            else:
                print(f"  Epoch {epoch+1:02d}/{self.epochs_per_task:02d} | "
                      f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Time: {time_taken:.1f}s")

        # After training this task, compute Fisher and snapshot params for EWC
        if self.ewc_lambda > 0:
            print(f"  [EWC] Computing Fisher Information ({self.fisher_samples} samples)...")
            fisher = compute_fisher(self.model, train_loader, self.device, self.fisher_samples)
            params = snapshot_params(self.model)
            self.fisher_list.append(fisher)
            self.params_list.append(params)
            print(f"  [EWC] Stored Fisher + params for Task {task_id + 1}")

    def evaluate_all_tasks(self, current_task_id: int):
        self.model.eval()
        task_accuracies = []
        print(f"\n[Eval] Testing all tasks up to {current_task_id + 1}...")

        with torch.no_grad():
            for t in range(current_task_id + 1):
                val_loader = self.loader.get_task_val_dataloader(t)
                correct = 0
                total = 0

                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
                        outputs = self.model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                acc = correct / total if total > 0 else 0.0
                task_accuracies.append(acc)
                print(f"  -> Task {t+1} Accuracy: {acc:.4f}")

        self.accuracy_matrix.append(task_accuracies)

        # Save checkpoint
        ckpt_path = os.path.join(self.checkpoint_dir, f"model_task_{current_task_id+1}.pth")
        torch.save(self.model.state_dict(), ckpt_path)

    def compute_forgetting(self):
        """Compute average forgetting across all completed tasks."""
        if not self.accuracy_matrix:
            return 0.0

        num_tasks = len(self.accuracy_matrix)
        if num_tasks < 2:
            return 0.0

        forgetting = 0.0
        for j in range(num_tasks - 1):
            max_past_acc = max(self.accuracy_matrix[i][j] for i in range(j, num_tasks - 1))
            current_acc = self.accuracy_matrix[-1][j]
            forgetting += max_past_acc - current_acc

        return forgetting / (num_tasks - 1)
