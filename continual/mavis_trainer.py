"""
MAVIS Trainer (PyTorch)
=======================
The Continual Learning training loop specifically designed for Phase 4 & 5.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim


class MAVISTrainer:
    """
    Trains the complete MAVIS architecture.
    """
    def __init__(self, model, loader, task_gen, learning_rate, epochs_per_task,
                 checkpoint_dir, device=None):
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

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.accuracy_matrix = []

        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_task(self, task_id: int):
        print(f"\n[{self.device.upper()}] --- Training Task {task_id + 1} (MAVIS) ---")
        train_loader = self.loader.get_task_train_dataloader(task_id)

        # Same optimizer and scheduler as Phase 3 (proven to hit 77% peak)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.learning_rate, 
            steps_per_epoch=len(train_loader), 
            epochs=self.epochs_per_task
        )
        scaler = torch.amp.GradScaler('cuda', enabled=(self.device == "cuda"))

        # 1. Train the Network
        for epoch in range(self.epochs_per_task):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            start_time = time.time()

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
                    # Offset targets to absolute class indices
                    start_cls = task_id * self.model.memory_bank.classes_per_task
                    end_cls = start_cls + self.model.memory_bank.classes_per_task
                    targets = targets + start_cls

                    # Pass task_id so model only retrieves PAST memories
                    logits = self.model(inputs, task_id=task_id)
                    
                    # Task Masking (Zero out gradients for non-current classes)
                    mask = torch.full_like(logits, -1e9) # Use -1e9 instead of -inf to avoid AMP NaN
                    mask[:, start_cls:end_cls] = logits[:, start_cls:end_cls]
                    
                    loss = self.criterion(mask, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += loss.item() * inputs.size(0)
                # Accuracy calculation based on unmasked logits
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            epoch_loss = total_loss / total
            epoch_acc = correct / total
            time_taken = time.time() - start_time

            print(f"  Epoch {epoch+1:02d}/{self.epochs_per_task:02d} | "
                  f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Time: {time_taken:.1f}s")

        # 2. Phase 4 Magic: Save Prototypes into Memory Bank
        print(f"  [MAVIS] Committing features for Task {task_id + 1} to Memory Bank...")
        # Note: We pass the train_loader so prototypes are built ONLY from training data (No Leakage)
        self.model.memory_bank.update_memory(task_id, train_loader, self.model.encoder, self.device)
        print(f"  [MAVIS] Memory successfully locked.")

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
                    # Offset targets to absolute class indices
                    start_cls = t * self.model.memory_bank.classes_per_task
                    targets = targets + start_cls
                    
                    with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
                        # No task_id passed during evaluation (Model retrieves all memories)
                        logits = self.model(inputs)
                        
                    _, predicted = logits.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                acc = correct / total if total > 0 else 0.0
                task_accuracies.append(acc)
                print(f"  -> Task {t+1} Accuracy: {acc:.4f}")

        self.accuracy_matrix.append(task_accuracies)

        # Save checkpoint
        ckpt_path = os.path.join(self.checkpoint_dir, f"mavis_task_{current_task_id+1}.pth")
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
