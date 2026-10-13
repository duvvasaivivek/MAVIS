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
                 checkpoint_dir, device=None, distill_lambda: float = 2.0, proto_lambda: float = 0.5, repel_lambda: float = 2.0,
                 lr_backbone_factor: float = 0.35, spatial_distill_lambda: float = 1.0, label_smoothing: float = 0.02,
                 proto_replay_lambda: float = 1.0):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = model.to(self.device)
        self.loader = loader
        self.task_gen = task_gen
        self.learning_rate = learning_rate
        self.lr_backbone_factor = lr_backbone_factor
        self.epochs_per_task = epochs_per_task
        self.checkpoint_dir = checkpoint_dir
        self.distill_lambda = distill_lambda
        self.spatial_distill_lambda = spatial_distill_lambda
        self.proto_lambda = proto_lambda
        self.repel_lambda = repel_lambda
        self.proto_replay_lambda = proto_replay_lambda
        self.old_encoder = None
        self.base_encoder = None  # Task 1 base coordinate anchor (E_0)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.accuracy_matrix = []       # Primary Continual Learning matrix (Hierarchical)
        self.accuracy_matrix_hier = []  # Measure 3: Hierarchical Memory-Routed Matrix
        self.accuracy_matrix_ens = []   # Fused Ensemble Matrix
        self.accuracy_matrix_ncm = []   # NCM Memory Matrix
        self.accuracy_matrix_head = []  # Aligned Linear Head Matrix

        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_task(self, task_id: int):
        print(f"\n[{self.device.upper()}] --- Training Task {task_id + 1} (MAVIS) ---", flush=True)
        train_loader = self.loader.get_task_train_dataloader(task_id)

        # Differential Learning Rates:
        # Task 1: Learn entire network from scratch at full learning_rate (high plasticity)
        # Tasks 2+: Damp backbone LR by lr_backbone_factor (stability), while head/attention have full plasticity
        if task_id == 0:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        else:
            optimizer = optim.AdamW([
                {'params': self.model.encoder.parameters(), 'lr': self.learning_rate * self.lr_backbone_factor},
                {'params': self.model.attention.parameters(), 'lr': self.learning_rate},
                {'params': self.model.fusion_gate.parameters(), 'lr': self.learning_rate},
                {'params': self.model.classifier.parameters(), 'lr': self.learning_rate}
            ], weight_decay=1e-2)

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=[p['lr'] for p in optimizer.param_groups], 
            steps_per_epoch=len(train_loader), 
            epochs=self.epochs_per_task,
            pct_start=0.15,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=100.0
        )
        scaler = torch.amp.GradScaler('cuda', enabled=(self.device == "cuda"))

        # 1. Train the Network
        for epoch in range(self.epochs_per_task):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            start_time = time.time()

            # Progressive Distillation Annealing (Warmup):
            # In incremental tasks (task_id > 0), allow high plasticity during early epochs (0-7)
            # so the backbone rapidly captures new class representations without fighting rigid distillation.
            # Ramps from 0.2 -> 1.0 linearly across the first 8 epochs.
            distill_scale = min(1.0, 0.2 + 0.8 * (epoch / 8.0)) if task_id > 0 else 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
                    # Offset targets to absolute class indices
                    start_cls = task_id * self.model.memory_bank.classes_per_task
                    end_cls = start_cls + self.model.memory_bank.classes_per_task
                    targets = targets + start_cls
                    local_targets = targets - start_cls

                    # Pass task_id so model retrieves past memories + current running prototypes, and get spatial features
                    logits, current_features, fused_features, spatials = self.model(
                        inputs, task_id=task_id, return_spatials=True
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
                            old_features, old_spatials = self.old_encoder(inputs, return_spatial=True)
                        loss_step = (1.0 - F.cosine_similarity(current_features, old_features, dim=-1)).mean()

                        if self.base_encoder is not None:
                            with torch.no_grad():
                                base_features = self.base_encoder(inputs)
                            loss_base = (1.0 - F.cosine_similarity(current_features, base_features, dim=-1)).mean()
                            loss_distill = 0.5 * loss_step + 0.5 * loss_base
                        else:
                            loss_distill = loss_step

                        # Multi-Axis PODNet spatial distillation (width & height slice pooling)
                        loss_spatial = 0.0
                        for s_curr, s_old in zip(spatials, old_spatials):
                            w_curr = F.normalize(s_curr.mean(dim=2).flatten(1), p=2, dim=-1)
                            w_old = F.normalize(s_old.mean(dim=2).flatten(1), p=2, dim=-1)
                            h_curr = F.normalize(s_curr.mean(dim=3).flatten(1), p=2, dim=-1)
                            h_old = F.normalize(s_old.mean(dim=3).flatten(1), p=2, dim=-1)
                            loss_w = (1.0 - (w_curr * w_old).sum(dim=-1)).mean()
                            loss_h = (1.0 - (h_curr * h_old).sum(dim=-1)).mean()
                            loss_spatial = loss_spatial + 0.5 * (loss_w + loss_h)
                        loss_spatial = loss_spatial / max(1, len(spatials))
                    else:
                        loss_distill = torch.tensor(0.0, device=self.device)
                        loss_spatial = torch.tensor(0.0, device=self.device)

                    # Contrastive Prototype Alignment Loss (InfoNCE on active task prototypes)
                    # Pulls features toward true prototype, actively repelling from other active prototypes
                    curr_protos = self.model.memory_bank.get_current_task_prototypes()
                    valid_proto_count = (self.model.memory_bank.running_counts > 0).sum()
                    if valid_proto_count >= 2:
                        norm_f = F.normalize(current_features, p=2, dim=-1)
                        norm_p = F.normalize(curr_protos, p=2, dim=-1).detach()
                        proto_logits = torch.matmul(norm_f, norm_p.t()) / 0.20
                        loss_proto = F.cross_entropy(proto_logits, local_targets)
                    else:
                        loss_proto = torch.tensor(0.0, device=self.device)

                    # Historical Memory Contrastive Repulsion + Attention Consistency
                    past_protos = self.model.memory_bank.get_past_memories(task_id)
                    if past_protos is not None and past_protos.size(0) > 0:
                        norm_curr = F.normalize(current_features, p=2, dim=-1)
                        norm_past = F.normalize(past_protos, p=2, dim=-1).detach()
                        sim_to_past = torch.matmul(norm_curr, norm_past.t())
                        loss_repel = F.relu(sim_to_past - 0.20).pow(2).mean()

                        # Attention Self-Reconstruction Loss (Attention fidelity on past memories)
                        recon_out = self.model.attention(norm_past, past_protos)
                        loss_recon = F.mse_loss(recon_out, norm_past)
                    else:
                        loss_repel = torch.tensor(0.0, device=self.device)
                        loss_recon = torch.tensor(0.0, device=self.device)

                    # Upgrade Measure 2: Gaussian Feature Covariance Replay on Classifier Head
                    past_protos_replay, past_labels_replay, past_vars_replay = self.model.memory_bank.get_past_prototypes_and_labels(task_id)
                    if past_protos_replay is not None and past_protos_replay.size(0) > 0:
                        curr_protos = self.model.memory_bank.get_current_task_prototypes()
                        valid_curr = (self.model.memory_bank.running_counts > 0)
                        if valid_curr.any():
                            curr_labels = torch.arange(start_cls, end_cls, device=self.device)[valid_curr]
                            curr_selected = curr_protos[valid_curr]
                            curr_vars = torch.full_like(curr_selected, 0.01)  # small default variance for live task
                            all_replay_protos = torch.cat([past_protos_replay, curr_selected], dim=0)
                            all_replay_labels = torch.cat([past_labels_replay, curr_labels], dim=0)
                            all_replay_vars = torch.cat([past_vars_replay, curr_vars], dim=0)
                        else:
                            all_replay_protos = past_protos_replay
                            all_replay_labels = past_labels_replay
                            all_replay_vars = past_vars_replay

                        # Gaussian covariance sampling along true feature variance ellipsoids
                        proto_noise = torch.randn_like(all_replay_protos) * (torch.sqrt(all_replay_vars) * 0.5)
                        replay_inputs = F.normalize(all_replay_protos + proto_noise, p=2, dim=-1)
                        replay_logits = self.model.classifier(replay_inputs)
                        loss_proto_replay = F.cross_entropy(replay_logits[:, :end_cls], all_replay_labels)
                    else:
                        loss_proto_replay = torch.tensor(0.0, device=self.device)

                    loss = (loss_ce 
                            + (self.distill_lambda * distill_scale) * loss_distill 
                            + (self.spatial_distill_lambda * distill_scale) * loss_spatial
                            + self.proto_lambda * loss_proto 
                            + self.repel_lambda * loss_repel
                            + 1.0 * loss_recon
                            + self.proto_replay_lambda * loss_proto_replay)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += loss.item() * inputs.size(0)
                # Accuracy calculation based on all seen logits
                _, predicted = logits[:, :end_cls].max(1)
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
        task_hier_accuracies = []
        task_routing_accuracies = []
        print(f"\n[Eval] Testing all tasks up to {current_task_id + 1}...", flush=True)

        seen_classes = (current_task_id + 1) * self.model.memory_bank.classes_per_task
        classes_per_task = self.model.memory_bank.classes_per_task
        num_tasks_seen = current_task_id + 1

        with torch.no_grad():
            for t in range(current_task_id + 1):
                val_loader = self.loader.get_task_val_dataloader(t)
                correct_head = 0
                correct_ncm = 0
                correct_ens = 0
                correct_hier = 0
                correct_task_routing = 0
                total = 0

                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    # Offset targets to absolute class indices
                    start_cls = t * classes_per_task
                    targets = targets + start_cls
                    
                    with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
                        # No task_id passed during evaluation (Model retrieves all memories)
                        logits, current_features, fused_features = self.model(
                            inputs, return_features=True
                        )
                        
                    # 1. Aligned Classifier Head Prediction (Flat across seen classes)
                    _, pred_head = logits[:, :seen_classes].max(1)
                    total += targets.size(0)
                    correct_head += pred_head.eq(targets).sum().item()

                    # 2. Multi-Prototype Nearest Class Mean (NCM) Prediction in prototype space (Flat)
                    ncm_scores = self.model.memory_bank.compute_ncm_scores(
                        current_features, up_to_task_id=current_task_id
                    )
                    if ncm_scores is not None:
                        _, pred_ncm = ncm_scores.max(1)
                        correct_ncm += pred_ncm.eq(targets).sum().item()

                        # 3. Ensemble Prediction (Flat: Head Logits + Scaled NCM Cosine Scores)
                        ens_scores = logits[:, :seen_classes] + self.model.classifier.fc.scale * ncm_scores
                        _, pred_ens = ens_scores.max(1)
                        correct_ens += pred_ens.eq(targets).sum().item()

                        # 4. Measure 3 Upgrade: Bayesian Soft Multi-Task Mixture Routing
                        # Stage 1: Compute task affinity scores from Memory Bank prototypes
                        task_affinities = []
                        for k in range(num_tasks_seen):
                            k_start = k * classes_per_task
                            k_end = k_start + classes_per_task
                            task_affinities.append(ncm_scores[:, k_start:k_end].max(dim=1)[0])
                        task_affinities = torch.stack(task_affinities, dim=1)  # [B, num_tasks_seen]
                        pred_task = task_affinities.argmax(dim=1)  # [B]
                        true_task = targets // classes_per_task
                        correct_task_routing += pred_task.eq(true_task).sum().item()

                        # Stage 2: Bayesian soft mixture across candidate tasks: P(c) = P(Task k) * P(c | Task k)
                        task_weights = F.softmax(task_affinities / 0.10, dim=1)  # [B, num_tasks_seen]
                        hier_scores = torch.zeros(inputs.size(0), seen_classes, device=self.device)
                        for k in range(num_tasks_seen):
                            k_start = k * classes_per_task
                            k_end = k_start + classes_per_task
                            intra_probs = F.softmax(ens_scores[:, k_start:k_end], dim=1)  # [B, classes_per_task]
                            hier_scores[:, k_start:k_end] = task_weights[:, k:k+1] * intra_probs

                        _, pred_hier = hier_scores.max(1)
                        correct_hier += pred_hier.eq(targets).sum().item()
                    else:
                        correct_ncm += correct_head
                        correct_ens += correct_head
                        correct_hier += correct_head
                        correct_task_routing += targets.size(0)

                head_acc = correct_head / total if total > 0 else 0.0
                ncm_acc = correct_ncm / total if total > 0 else 0.0
                ens_acc = correct_ens / total if total > 0 else 0.0
                hier_acc = correct_hier / total if total > 0 else 0.0
                routing_acc = correct_task_routing / total if total > 0 else 0.0

                task_head_accuracies.append(head_acc)
                task_ncm_accuracies.append(ncm_acc)
                task_ens_accuracies.append(ens_acc)
                task_hier_accuracies.append(hier_acc)
                task_routing_accuracies.append(routing_acc)

                print(f"  -> Task {t+1} Hierarchical: {hier_acc:.4f} | Ensemble: {ens_acc:.4f} | NCM: {ncm_acc:.4f} | Head: {head_acc:.4f} (Task ID Acc: {routing_acc:.4f})", flush=True)

        # Store primary matrix as Hierarchical (Measure 3)
        self.accuracy_matrix.append(task_hier_accuracies)
        self.accuracy_matrix_hier.append(task_hier_accuracies)
        self.accuracy_matrix_ens.append(task_ens_accuracies)
        self.accuracy_matrix_ncm.append(task_ncm_accuracies)
        self.accuracy_matrix_head.append(task_head_accuracies)

        # Save checkpoint (model weights + accuracy history)
        ckpt_path = os.path.join(self.checkpoint_dir, f"mavis_task_{current_task_id+1}.pth")
        torch.save(self.model.state_dict(), ckpt_path)
        metrics_ckpt = os.path.join(self.checkpoint_dir, "accuracy_matrix.json")
        metrics_hier_ckpt = os.path.join(self.checkpoint_dir, "accuracy_matrix_hier.json")
        metrics_ens_ckpt = os.path.join(self.checkpoint_dir, "accuracy_matrix_ens.json")
        metrics_ncm_ckpt = os.path.join(self.checkpoint_dir, "accuracy_matrix_ncm.json")
        metrics_head_ckpt = os.path.join(self.checkpoint_dir, "accuracy_matrix_head.json")
        import json
        with open(metrics_ckpt, "w") as f:
            json.dump(self.accuracy_matrix, f, indent=2)
        with open(metrics_hier_ckpt, "w") as f:
            json.dump(self.accuracy_matrix_hier, f, indent=2)
        with open(metrics_ens_ckpt, "w") as f:
            json.dump(self.accuracy_matrix_ens, f, indent=2)
        with open(metrics_ncm_ckpt, "w") as f:
            json.dump(self.accuracy_matrix_ncm, f, indent=2)
        with open(metrics_head_ckpt, "w") as f:
            json.dump(self.accuracy_matrix_head, f, indent=2)

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
