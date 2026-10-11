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
                 checkpoint_dir, device=None, distill_lambda: float = 4.0, proto_lambda: float = 1.0, repel_lambda: float = 2.0):
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
        self.repel_lambda = repel_lambda
        self.old_encoder = None
        self.base_encoder = None  # Task 1 base coordinate anchor (E_0)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.accuracy_matrix = []       # Primary Continual Learning matrix (NCM)
        self.accuracy_matrix_ncm = []   # NCM Memory Matrix
        self.accuracy_matrix_head = []  # Aligned Linear Head Matrix
        self.accuracy_matrix_ens = []   # Fused Ensemble Matrix

        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_task(self, task_id: int):
        print(f"\n[{self.device.upper()}] --- Training Task {task_id + 1} (MAVIS) ---", flush=True)
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

                    # Multi-Anchor Distillation: Anchors to BOTH Task 1 Base E_0 AND Step E_{t-1}
                    if self.old_encoder is not None:
                        with torch.no_grad():
                            old_features = self.old_encoder(inputs)
                        loss_step = (1.0 - F.cosine_similarity(current_features, old_features, dim=-1)).mean()

                        if self.base_encoder is not None:
                            with torch.no_grad():
                                base_features = self.base_encoder(inputs)
                            loss_base = (1.0 - F.cosine_similarity(current_features, base_features, dim=-1)).mean()
                            loss_distill = 0.5 * loss_step + 0.5 * loss_base
                        else:
                            loss_distill = loss_step
                    else:
                        loss_distill = torch.tensor(0.0, device=self.device)

                    # Prototype Alignment Loss (Clusters current features around class prototypes)
                    curr_protos = self.model.memory_bank.get_current_task_prototypes()
                    target_protos = curr_protos[local_targets]
                    valid_proto_mask = self.model.memory_bank.running_counts[local_targets] > 0
                    if valid_proto_mask.any():
                        norm_f = F.normalize(current_features[valid_proto_mask], p=2, dim=-1)
                        norm_p = target_protos[valid_proto_mask].detach()
                        loss_proto = (1.0 - (norm_f * norm_p).sum(dim=-1)).mean()
                    else:
                        loss_proto = torch.tensor(0.0, device=self.device)

                    # Historical Memory Contrastive Repulsion (Pushes new features away from past prototypes)
                    past_protos = self.model.memory_bank.get_past_memories(task_id)
                    if past_protos is not None:
                        norm_curr = F.normalize(current_features, p=2, dim=-1)
                        norm_past = F.normalize(past_protos, p=2, dim=-1).detach()
                        sim_to_past = torch.matmul(norm_curr, norm_past.t())
                        loss_repel = F.relu(sim_to_past - 0.20).pow(2).mean()
                    else:
                        loss_repel = torch.tensor(0.0, device=self.device)

                    loss = (loss_ce 
                            + self.distill_lambda * loss_distill 
                            + self.proto_lambda * loss_proto 
                            + self.repel_lambda * loss_repel)

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
                  f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Time: {time_taken:.1f}s", flush=True)

        # 2. Phase 4: Save Prototypes into Memory Bank (Multi-Prototype K-Means, Lever 1)
        print(f"  [MAVIS] Committing features for Task {task_id + 1} to Memory Bank (K={self.model.memory_bank.prototypes_per_class})...", flush=True)
        # Note: We pass the train_loader so prototypes are built ONLY from training data (No Leakage)
        self.model.memory_bank.update_memory(task_id, train_loader, self.model.encoder, self.device)
        print(f"  [MAVIS] Memory successfully locked.", flush=True)

        # Lever 2: Balance Classifier Head via Weight Aligning (WA - CVPR 2020)
        num_old_classes = task_id * self.model.memory_bank.classes_per_task
        num_new_classes = self.model.memory_bank.classes_per_task
        if num_old_classes > 0:
            all_mean_protos = self.model.memory_bank.get_mean_class_prototypes(up_to_task_id=task_id)
            self.model.classifier.align_weights(num_old_classes, num_new_classes, prototypes=all_mean_protos)

        # 3. Snapshot Base Anchor E_0 on Task 1 completion (Grounding Anchor for all future tasks)
        if self.base_encoder is None and task_id == 0:
            self.base_encoder = copy.deepcopy(self.model.encoder)
            self.base_encoder.eval()
            for p in self.base_encoder.parameters():
                p.requires_grad = False
            print("  [MAVIS] Snapshotted Base Anchor Encoder E_0 for cross-task coordinate grounding.", flush=True)

        # 4. Snapshot step encoder for next task representation preservation
        self.old_encoder = copy.deepcopy(self.model.encoder)
        self.old_encoder.eval()
        for p in self.old_encoder.parameters():
            p.requires_grad = False

    def evaluate_all_tasks(self, current_task_id: int):
        self.model.eval()
        task_head_accuracies = []
        task_ncm_accuracies = []
        task_ens_accuracies = []
        print(f"\n[Eval] Testing all tasks up to {current_task_id + 1}...", flush=True)

        with torch.no_grad():
            for t in range(current_task_id + 1):
                val_loader = self.loader.get_task_val_dataloader(t)
                correct_head = 0
                correct_ncm = 0
                correct_ens = 0
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
                    _, pred_head = logits.max(1)
                    total += targets.size(0)
                    correct_head += pred_head.eq(targets).sum().item()

                    # 2. Multi-Prototype Nearest Class Mean (NCM) Prediction in prototype space
                    ncm_scores = self.model.memory_bank.compute_ncm_scores(
                        current_features, up_to_task_id=current_task_id
                    )
                    if ncm_scores is not None:
                        _, pred_ncm = ncm_scores.max(1)
                        correct_ncm += pred_ncm.eq(targets).sum().item()

                        # 3. Ensemble Prediction (Head Logits + Scaled NCM Cosine Scores)
                        seen_classes = (current_task_id + 1) * self.model.memory_bank.classes_per_task
                        ens_scores = logits[:, :seen_classes] + self.model.classifier.fc.scale * ncm_scores
                        _, pred_ens = ens_scores.max(1)
                        correct_ens += pred_ens.eq(targets).sum().item()
                    else:
                        correct_ncm += correct_head
                        correct_ens += correct_head

                head_acc = correct_head / total if total > 0 else 0.0
                ncm_acc = correct_ncm / total if total > 0 else 0.0
                ens_acc = correct_ens / total if total > 0 else 0.0

                task_head_accuracies.append(head_acc)
                task_ncm_accuracies.append(ncm_acc)
                task_ens_accuracies.append(ens_acc)

                print(f"  -> Task {t+1} NCM: {ncm_acc:.4f} (Memory) | Ensemble: {ens_acc:.4f} | Head: {head_acc:.4f}", flush=True)

        # Store primary matrix as NCM (the robust continual memory representation)
        self.accuracy_matrix.append(task_ncm_accuracies)
        self.accuracy_matrix_ncm.append(task_ncm_accuracies)
        self.accuracy_matrix_head.append(task_head_accuracies)
        self.accuracy_matrix_ens.append(task_ens_accuracies)

        # Save checkpoint (model weights + accuracy history)
        ckpt_path = os.path.join(self.checkpoint_dir, f"mavis_task_{current_task_id+1}.pth")
        torch.save(self.model.state_dict(), ckpt_path)
        metrics_ckpt = os.path.join(self.checkpoint_dir, "accuracy_matrix.json")
        metrics_ncm_ckpt = os.path.join(self.checkpoint_dir, "accuracy_matrix_ncm.json")
        metrics_head_ckpt = os.path.join(self.checkpoint_dir, "accuracy_matrix_head.json")
        metrics_ens_ckpt = os.path.join(self.checkpoint_dir, "accuracy_matrix_ens.json")
        import json
        with open(metrics_ckpt, "w") as f:
            json.dump(self.accuracy_matrix, f, indent=2)
        with open(metrics_ncm_ckpt, "w") as f:
            json.dump(self.accuracy_matrix_ncm, f, indent=2)
        with open(metrics_head_ckpt, "w") as f:
            json.dump(self.accuracy_matrix_head, f, indent=2)
        with open(metrics_ens_ckpt, "w") as f:
            json.dump(self.accuracy_matrix_ens, f, indent=2)

    def compute_forgetting(self, matrix=None):
        """Compute average forgetting across all completed tasks."""
        target_matrix = matrix if matrix is not None else self.accuracy_matrix
        if not target_matrix:
            return 0.0

        num_tasks = len(target_matrix)
        if num_tasks < 2:
            return 0.0

        forgetting = 0.0
        for j in range(num_tasks - 1):
            max_past_acc = max(target_matrix[i][j] for i in range(j, num_tasks - 1))
            current_acc = target_matrix[-1][j]
            forgetting += max_past_acc - current_acc

        return forgetting / (num_tasks - 1)
