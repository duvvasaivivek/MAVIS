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
                 checkpoint_dir, device=None, distill_lambda: float = 4.0, proto_lambda: float = 1.0):
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
        self.proto_lambda = proto_lambda
        self.old_encoder = None

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.accuracy_matrix = []
        self.accuracy_matrix_ncm = []

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
                    local_targets = targets - start_cls

                    # Pass task_id so model retrieves past memories + current running prototypes
                    logits, current_features, fused_features = self.model(
                        inputs, task_id=task_id, return_features=True
                    )
                    
                    # Update online running prototypes for active task (no grad)
                    self.model.memory_bank.update_running_prototypes(
                        current_features, local_targets
                    )

                    # Task Masking (Only compute CE loss on the neurons active for this task)
                    active_logits = logits[:, start_cls:end_cls]
                    loss_ce = self.criterion(active_logits, local_targets)

                    # Lever 4: Representation distillation loss (anchored to previous task)
                    if self.old_encoder is not None:
                        with torch.no_grad():
                            old_features = self.old_encoder(inputs)
                        loss_distill = (1.0 - F.cosine_similarity(current_features, old_features, dim=-1)).mean()
                    else:
                        loss_distill = torch.tensor(0.0, device=self.device)

                    # Lever 4: Prototype Alignment Loss (prevents drift and clusters features around prototypes)
                    curr_protos = self.model.memory_bank.get_current_task_prototypes()
                    target_protos = curr_protos[local_targets]
                    valid_proto_mask = self.model.memory_bank.running_counts[local_targets] > 0
                    if valid_proto_mask.any():
                        norm_f = F.normalize(current_features[valid_proto_mask], p=2, dim=-1)
                        norm_p = target_protos[valid_proto_mask].detach()
                        loss_proto = (1.0 - (norm_f * norm_p).sum(dim=-1)).mean()
                    else:
                        loss_proto = torch.tensor(0.0, device=self.device)

                    loss = loss_ce + self.distill_lambda * loss_distill + self.proto_lambda * loss_proto

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

        # 2. Phase 4: Save Prototypes into Memory Bank (Multi-Prototype K-Means, Lever 1)
        print(f"  [MAVIS] Committing features for Task {task_id + 1} to Memory Bank (K={self.model.memory_bank.prototypes_per_class})...")
        # Note: We pass the train_loader so prototypes are built ONLY from training data (No Leakage)
        self.model.memory_bank.update_memory(task_id, train_loader, self.model.encoder, self.device)
        print(f"  [MAVIS] Memory successfully locked.")

        # Lever 2: Balance Classifier Head via Weight Aligning (WA - CVPR 2020)
        num_old_classes = task_id * self.model.memory_bank.classes_per_task
        num_new_classes = self.model.memory_bank.classes_per_task
        if num_old_classes > 0:
            all_mean_protos = self.model.memory_bank.get_mean_class_prototypes(up_to_task_id=task_id)
            self.model.classifier.align_weights(num_old_classes, num_new_classes, prototypes=all_mean_protos)

        # 3. Snapshot encoder for next task representation preservation
        self.old_encoder = copy.deepcopy(self.model.encoder)
        self.old_encoder.eval()
        for p in self.old_encoder.parameters():
            p.requires_grad = False

    def evaluate_all_tasks(self, current_task_id: int):
        self.model.eval()
        task_accuracies = []
        task_ncm_accuracies = []
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
                        
                    # 1. Aligned Classifier Head Prediction
                    _, predicted = logits.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    # 2. Multi-Prototype Nearest Class Mean (NCM) Prediction (Lever 1)
                    ncm_scores = self.model.memory_bank.compute_ncm_scores(
                        fused_features, up_to_task_id=current_task_id
                    )
                    if ncm_scores is not None:
                        _, ncm_pred = ncm_scores.max(1)
                        correct_ncm += ncm_pred.eq(targets).sum().item()

                acc = correct / total if total > 0 else 0.0
                task_accuracies.append(acc)
                if total > 0:
                    ncm_acc = correct_ncm / total
                    task_ncm_accuracies.append(ncm_acc)
                    print(f"  -> Task {t+1} Accuracy: {acc:.4f} (WA Head) | NCM: {ncm_acc:.4f} (Memory)")
                else:
                    print(f"  -> Task {t+1} Accuracy: {acc:.4f}")

        self.accuracy_matrix.append(task_accuracies)
        self.accuracy_matrix_ncm.append(task_ncm_accuracies)

        # Save checkpoint (model weights + accuracy history)
        ckpt_path = os.path.join(self.checkpoint_dir, f"mavis_task_{current_task_id+1}.pth")
        torch.save(self.model.state_dict(), ckpt_path)
        metrics_ckpt = os.path.join(self.checkpoint_dir, "accuracy_matrix.json")
        metrics_ncm_ckpt = os.path.join(self.checkpoint_dir, "accuracy_matrix_ncm.json")
        import json
        with open(metrics_ckpt, "w") as f:
            json.dump(self.accuracy_matrix, f, indent=2)
        with open(metrics_ncm_ckpt, "w") as f:
            json.dump(self.accuracy_matrix_ncm, f, indent=2)

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
