"""
MAVIS Trainer (PyTorch)
=======================
The Continual Learning training loop specifically designed for Phase 4 & 5.
"""

import os
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MAVISTrainer:
    """
    Trains the complete MAVIS architecture with Cosine Linear classification,
    representation-preserving feature distillation, and Memory Attention retrieval.
    """
    def __init__(self, model, loader, task_gen, learning_rate, epochs_per_task,
                 checkpoint_dir, device=None, distill_lambda: float = 2.0):
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
        self.distill_lambda = distill_lambda
        self.old_encoder = None

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.accuracy_matrix = []

        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_task(self, task_id: int):
        print(f"\n[{self.device.upper()}] --- Training Task {task_id + 1} (MAVIS) ---")
        train_loader = self.loader.get_task_train_dataloader(task_id)

        # AdamW optimizer and OneCycleLR scheduler
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

                    # Pass task_id so model retrieves past memories + current running prototypes
                    logits, current_features, fused_features = self.model(
                        inputs, task_id=task_id, return_features=True
                    )
                    
                    # Update online running prototypes for active task (no grad)
                    self.model.memory_bank.update_running_prototypes(
                        current_features, targets - start_cls
                    )

                    # Task Masking (Only compute CE loss on the neurons active for this task)
                    active_logits = logits[:, start_cls:end_cls]
                    loss_ce = self.criterion(active_logits, targets - start_cls)

                    # Less-Forget representation distillation loss
                    if self.old_encoder is not None:
                        with torch.no_grad():
                            old_features = self.old_encoder(inputs)
                        loss_distill = (1.0 - F.cosine_similarity(current_features, old_features, dim=-1)).mean()
                        loss = loss_ce + self.distill_lambda * loss_distill
                    else:
                        loss = loss_ce

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

        # 2. Phase 4: Save Prototypes into Memory Bank
        print(f"  [MAVIS] Committing features for Task {task_id + 1} to Memory Bank...")
        # Note: We pass the train_loader so prototypes are built ONLY from training data (No Leakage)
        self.model.memory_bank.update_memory(task_id, train_loader, self.model.encoder, self.device)
        print(f"  [MAVIS] Memory successfully locked.")

        # 3. Snapshot encoder for next task representation preservation
        self.old_encoder = copy.deepcopy(self.model.encoder)
        self.old_encoder.eval()
        for p in self.old_encoder.parameters():
            p.requires_grad = False

    def evaluate_all_tasks(self, current_task_id: int):
        self.model.eval()
        task_accuracies = []
        all_prototypes = self.model.memory_bank.get_all_populated_memories()
        print(f"\n[Eval] Testing all tasks up to {current_task_id + 1}...")

        with torch.no_grad():
            for t in range(current_task_id + 1):
                val_loader = self.loader.get_task_val_dataloader(t)
                correct = 0
                correct_ncm = 0
                total = 0

                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    # Offset targets to absolute class indices
                    start_cls = t * self.model.memory_bank.classes_per_task
                    targets = targets + start_cls
                    
                    with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
                        # No task_id passed during evaluation (Model retrieves all memories)
                        logits, current_features, fused_features = self.model(
                            inputs, return_features=True
                        )
                        
                    _, predicted = logits.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    if all_prototypes is not None:
                        ncm_scores = F.linear(F.normalize(fused_features, p=2, dim=1), all_prototypes)
                        _, ncm_pred = ncm_scores.max(1)
                        correct_ncm += ncm_pred.eq(targets).sum().item()

                acc = correct / total if total > 0 else 0.0
                task_accuracies.append(acc)
                if all_prototypes is not None and total > 0:
                    ncm_acc = correct_ncm / total
                    print(f"  -> Task {t+1} Accuracy: {acc:.4f} (NCM: {ncm_acc:.4f})")
                else:
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
