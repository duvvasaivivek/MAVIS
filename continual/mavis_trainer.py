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

from continual.ewc import OnlineEWC


class MAVISTrainer:
    """
    Trains the complete MAVIS architecture with Cosine Linear classification,
    representation-preserving feature distillation, Memory Attention retrieval,
    and parameter-level Elastic Weight Consolidation (Phase 5 Full MAVIS).
    """
    def __init__(self, model, loader, task_gen, learning_rate, epochs_per_task,
                 checkpoint_dir, device=None, distill_lambda: float = 3.0, proto_lambda: float = 0.5, repel_lambda: float = 2.0,
                 lr_backbone_factor: float = 0.20, spatial_distill_lambda: float = 1.5, label_smoothing: float = 0.02,
                 proto_replay_lambda: float = 2.0, ewc_lambda: float = 5000.0, fisher_samples: int = 2000,
                 gamma: float = 0.95, proto_finetune_epochs: int = 25, proto_finetune_draws: int = 10):
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
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples
        self.gamma = gamma
        self.proto_finetune_epochs = proto_finetune_epochs
        self.proto_finetune_draws = proto_finetune_draws
        self.old_encoder = None
        self.base_encoder = None  # Task 1 base coordinate anchor (E_0)

        # Phase 5: Online EWC module for parameter curvature preservation
        self.ewc_module = OnlineEWC(model=self.model, ewc_lambda=self.ewc_lambda, gamma=self.gamma) if self.ewc_lambda > 0 else None

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.accuracy_matrix = []            # Primary Continual Learning matrix (Hierarchical)
        self.accuracy_matrix_hier = []       # Measure 3: Hierarchical Memory-Routed Matrix
        self.accuracy_matrix_ens = []        # Fused Ensemble Matrix (Top-1)
        self.accuracy_matrix_ncm = []        # NCM Memory Matrix
        self.accuracy_matrix_head = []       # Aligned Linear Head Matrix (Top-1)
        self.accuracy_matrix_top5_ens = []   # Fused Ensemble Matrix (Top-5)
        self.accuracy_matrix_top5_head = []  # Aligned Linear Head Matrix (Top-5)
        self.accuracy_matrix_top5_hier = []  # Hierarchical Matrix (Top-5)

        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_task(self, task_id: int):
        mode_str = f"Full MAVIS (Memory + Attention + EWC λ={self.ewc_lambda})" if (self.ewc_lambda > 0) else "MAVIS (Memory + Attention)"
        print(f"\n[{self.device.upper()}] --- Training Task {task_id + 1} ({mode_str}) ---", flush=True)
        train_loader = self.loader.get_task_train_dataloader(task_id)

        # Differential Learning Rates:
        # Task 1: Learn entire network from scratch at full learning_rate (high plasticity)
        # Tasks 2+: Damp backbone LR by lr_backbone_factor (stability), while head/attention have full plasticity
        if task_id == 0:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        else:
            # Task-Adaptive Plasticity: Gently decays backbone LR as tasks accumulate (0.20 -> 0.12)
            # Preserves plastic feature acquisition early on while safeguarding earlier coordinates in late tasks
            effective_backbone_factor = max(0.12, self.lr_backbone_factor - 0.01 * (task_id - 1))
            optimizer = optim.AdamW([
                {'params': self.model.encoder.parameters(), 'lr': self.learning_rate * effective_backbone_factor},
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
            total_ewc = 0
            correct = 0
            task_correct = 0
            total = 0

            start_time = time.time()

            # Progressive Distillation Annealing (Warmup):
            # In incremental tasks (task_id > 0), apply strong distillation from early epochs
            # to constrain encoder drift while still allowing new-class adaptation.
            # Ramps from 0.5 -> 1.0 linearly across the first 5 epochs.
            distill_scale = min(1.0, 0.5 + 0.5 * (epoch / 5.0)) if task_id > 0 else 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
                    # Offset targets to absolute class indices
                    start_cls = task_id * self.model.memory_bank.classes_per_task
                    end_cls = start_cls + self.model.memory_bank.classes_per_task
                    targets = targets + start_cls
                    local_targets = targets - start_cls

                    # Pass task_id so model retrieves past memories + current running prototypes, targets for margin, and get spatial features
                    logits, current_features, fused_features, spatials = self.model(
                        inputs, task_id=task_id, targets=targets, return_spatials=True
                    )
                    
                    # Update online running prototypes for active task (no grad)
                    self.model.memory_bank.update_running_prototypes(
                        current_features, local_targets
                    )

                    # Task-Masked Cross-Entropy on live batch:
                    # Trains active task neurons on live images without applying destructive negative
                    # gradients to historical classes. Cross-task discrimination is handled by
                    # Gaussian Prototype Replay and post-task Balanced Finetuning.
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

                    # Phase 5: Online EWC Penalty (Parameter-Level Curvature Protection)
                    if self.ewc_module is not None and self.ewc_module.task_count > 0:
                        loss_ewc = self.ewc_module.penalty(self.model)
                        total_ewc += loss_ewc.item() * inputs.size(0)
                    else:
                        loss_ewc = torch.tensor(0.0, device=self.device)

                    loss = (loss_ce 
                            + (self.distill_lambda * distill_scale) * loss_distill 
                            + (self.spatial_distill_lambda * distill_scale) * loss_spatial
                            + self.proto_lambda * loss_proto 
                            + self.repel_lambda * loss_repel
                            + 1.0 * loss_recon
                            + self.proto_replay_lambda * loss_proto_replay
                            + loss_ewc)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += loss.item() * inputs.size(0)
                # 1. Intra-task accuracy (among the 20 classes of the active task)
                _, task_pred = logits[:, start_cls:end_cls].max(1)
                task_correct += task_pred.eq(local_targets).sum().item()

                # 2. Unified accuracy (among all seen classes up to current task)
                _, predicted = logits[:, :end_cls].max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            epoch_loss = total_loss / total
            epoch_acc = correct / total
            epoch_task_acc = task_correct / total
            time_taken = time.time() - start_time

            if self.ewc_module is not None and self.ewc_module.task_count > 0:
                epoch_ewc = total_ewc / total
                print(f"  Epoch {epoch+1:02d}/{self.epochs_per_task:02d} | "
                      f"Loss: {epoch_loss:.4f} | EWC: {epoch_ewc:.4f} | TaskAcc: {epoch_task_acc:.4f} | SeenAcc: {epoch_acc:.4f} | Time: {time_taken:.1f}s", flush=True)
            else:
                print(f"  Epoch {epoch+1:02d}/{self.epochs_per_task:02d} | "
                      f"Loss: {epoch_loss:.4f} | TaskAcc: {epoch_task_acc:.4f} | SeenAcc: {epoch_acc:.4f} | Time: {time_taken:.1f}s", flush=True)

        # 2. Phase 4: Save Prototypes into Memory Bank (Multi-Prototype K-Means, Lever 1)
        print(f"  [MAVIS] Committing features for Task {task_id + 1} to Memory Bank (K={self.model.memory_bank.prototypes_per_class})...", flush=True)
        # Note: We pass the train_loader so prototypes are built ONLY from training data (No Leakage)
        self.model.memory_bank.update_memory(task_id, train_loader, self.model.encoder, self.device)
        print(f"  [MAVIS] Memory successfully locked.", flush=True)

        # 2b. Prototype Re-Alignment: Re-encode ALL past tasks' prototypes through the current encoder.
        # This fixes the critical representation drift problem where stored prototypes become stale
        # as the encoder evolves across tasks, causing NCM and hierarchical classification to fail.
        if task_id > 0:
            print(f"  [MAVIS] Re-aligning {task_id} past task(s) prototypes through current encoder...", flush=True)
            for past_t in range(task_id):
                if self.model.memory_bank.is_populated[past_t]:
                    past_loader = self.loader.get_task_train_dataloader(past_t)
                    self.model.memory_bank.update_memory(past_t, past_loader, self.model.encoder, self.device)
            print(f"  [MAVIS] Prototype re-alignment complete.", flush=True)

        # Lever 2: Balance Classifier Head via Weight Aligning (WA - CVPR 2020)
        num_old_classes = task_id * self.model.memory_bank.classes_per_task
        num_new_classes = self.model.memory_bank.classes_per_task
        if num_old_classes > 0:
            all_mean_protos = self.model.memory_bank.get_mean_class_prototypes(up_to_task_id=task_id)
            self.model.classifier.align_weights(num_old_classes, num_new_classes, prototypes=all_mean_protos)

        # 2c. Balanced Classifier Finetuning (frozen encoder, prototype replay)
        # Re-calibrates the classifier head for ALL seen classes, preventing
        # the head from being dominated by the most recent task.
        if task_id > 0:
            print(f"  [MAVIS] Running Balanced Classifier Finetuning ({self.proto_finetune_epochs} epochs, {self.proto_finetune_draws} draws/proto)...", flush=True)
            self._balanced_classifier_finetune(
                task_id,
                num_epochs=self.proto_finetune_epochs,
                num_draws=self.proto_finetune_draws
            )

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

        # 5. Phase 5: Update Online EWC Fisher Information Matrix and parameter snapshot
        if self.ewc_module is not None:
            print(f"  [MAVIS EWC] Updating Online Fisher Information Matrix ({self.fisher_samples} samples, gamma={self.gamma})...", flush=True)
            self.ewc_module.update(
                self.model, train_loader, self.device,
                num_samples=self.fisher_samples,
                task_id=task_id,
                classes_per_task=self.model.memory_bank.classes_per_task
            )
            print(f"  [MAVIS EWC] Fisher Matrix and parameter anchor updated for Task {task_id + 1}.", flush=True)

    def _balanced_classifier_finetune(self, task_id: int, num_epochs: int = 25, num_draws: int = 10, ft_lr: float = 0.005):
        """
        Post-task balanced classifier finetuning with frozen encoder.
        Re-calibrates the classification head across ALL seen classes using
        Gaussian-augmented prototype replay, preventing recency bias.
        """
        end_cls = (task_id + 1) * self.model.memory_bank.classes_per_task

        # Collect prototypes, labels, and variances from ALL seen tasks
        all_protos = []
        all_labels = []
        all_vars = []
        for t in range(task_id + 1):
            if self.model.memory_bank.is_populated[t]:
                t_start = t * self.model.memory_bank.classes_per_task
                for c in range(self.model.memory_bank.classes_per_task):
                    protos_c = self.model.memory_bank.memory[t, c]  # [K, D]
                    var_c = self.model.memory_bank.class_variances[t, c]  # [D]
                    for k in range(self.model.memory_bank.prototypes_per_class):
                        all_protos.append(protos_c[k])
                        all_labels.append(t_start + c)
                        all_vars.append(var_c)

        if not all_protos:
            return

        all_protos = torch.stack(all_protos).to(self.device)
        all_labels = torch.tensor(all_labels, device=self.device)
        all_vars = torch.stack(all_vars).to(self.device)

        # Freeze encoder, attention, and fusion gate — strictly train classifier head
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        for p in self.model.attention.parameters():
            p.requires_grad = False
        for p in self.model.fusion_gate.parameters():
            p.requires_grad = False

        ft_optimizer = optim.AdamW(self.model.classifier.parameters(), lr=ft_lr, weight_decay=1e-4)
        ft_scheduler = optim.lr_scheduler.CosineAnnealingLR(ft_optimizer, T_max=num_epochs, eta_min=1e-5)

        final_acc = 0.0
        for epoch in range(num_epochs):
            perm = torch.randperm(all_protos.size(0))
            shuffled_p = all_protos[perm]
            shuffled_l = all_labels[perm]
            shuffled_v = all_vars[perm]

            total_correct = 0
            total_samples = 0
            total_loss = 0.0
            batch_size = 256

            for start in range(0, shuffled_p.size(0), batch_size):
                end = min(start + batch_size, shuffled_p.size(0))
                p_batch = shuffled_p[start:end]
                l_batch = shuffled_l[start:end]
                v_batch = shuffled_v[start:end]

                # Multi-sample Gaussian augmentation (num_draws noise draws per prototype)
                aug_inputs_list = []
                aug_labels_list = []
                for _ in range(num_draws):
                    noise = torch.randn_like(p_batch) * (torch.sqrt(torch.clamp(v_batch, min=1e-6)) * 0.5)
                    aug_inputs_list.append(F.normalize(p_batch + noise, p=2, dim=-1))
                    aug_labels_list.append(l_batch)
                aug_inputs = torch.cat(aug_inputs_list, dim=0)
                aug_labels = torch.cat(aug_labels_list, dim=0)

                # Feature Distribution Alignment: Harmonize prototype replays through Attention & Gated Fusion
                with torch.no_grad():
                    fused_aug_inputs = self.model.fuse_features(aug_inputs)

                ft_optimizer.zero_grad()
                # CRITICAL FIX: Pass targets=None so additive margin (m=0.20) does NOT artificially
                # subtract from the true class logit and distort post-hoc head calibration!
                logits = self.model.classifier(fused_aug_inputs, targets=None)
                loss = F.cross_entropy(logits[:, :end_cls], aug_labels)
                loss.backward()
                ft_optimizer.step()

                _, pred = logits[:, :end_cls].max(1)
                total_correct += pred.eq(aug_labels).sum().item()
                total_samples += aug_labels.size(0)
                total_loss += loss.item() * aug_labels.size(0)

            ft_scheduler.step()
            epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
            epoch_acc = total_correct / total_samples if total_samples > 0 else 0.0

            if (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == num_epochs:
                print(f"    [Finetune Head] Epoch {epoch+1:02d}/{num_epochs:02d} | Loss: {epoch_loss:.4f} | Balanced Proto Acc: {epoch_acc*100:.2f}%", flush=True)

        final_acc = epoch_acc

        # Unfreeze all modules for the next task
        for p in self.model.encoder.parameters():
            p.requires_grad = True
        for p in self.model.attention.parameters():
            p.requires_grad = True
        for p in self.model.fusion_gate.parameters():
            p.requires_grad = True

        print(f"  [MAVIS] Balanced classifier finetuning complete: final_acc={final_acc*100:.2f}%", flush=True)

    def evaluate_all_tasks(self, current_task_id: int):
        self.model.eval()
        task_head_accuracies = []
        task_head_top5_accuracies = []
        task_ncm_accuracies = []
        task_ens_accuracies = []
        task_ens_top5_accuracies = []
        task_hier_accuracies = []
        task_hier_top5_accuracies = []
        task_routing_accuracies = []
        print(f"\n[Eval] Testing all tasks up to {current_task_id + 1}...", flush=True)

        seen_classes = (current_task_id + 1) * self.model.memory_bank.classes_per_task
        classes_per_task = self.model.memory_bank.classes_per_task
        num_tasks_seen = current_task_id + 1

        def calc_topk_correct(scores, targets, k=5):
            k_val = min(k, scores.size(1))
            _, topk_preds = scores.topk(k_val, dim=1, largest=True, sorted=True)
            return topk_preds.eq(targets.unsqueeze(1)).any(dim=1).sum().item()

        with torch.no_grad():
            for t in range(current_task_id + 1):
                val_loader = self.loader.get_task_val_dataloader(t)
                correct_head = 0
                correct_head_top5 = 0
                correct_ncm = 0
                correct_ens = 0
                correct_ens_top5 = 0
                correct_hier = 0
                correct_hier_top5 = 0
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
                    head_scores = logits[:, :seen_classes]
                    _, pred_head = head_scores.max(1)
                    total += targets.size(0)
                    correct_head += pred_head.eq(targets).sum().item()
                    correct_head_top5 += calc_topk_correct(head_scores, targets, k=5)

                    # 2. Multi-Prototype Nearest Class Mean (NCM) Prediction in prototype space (Flat)
                    ncm_scores = self.model.memory_bank.compute_ncm_scores(
                        current_features, up_to_task_id=current_task_id
                    )
                    if ncm_scores is not None:
                        _, pred_ncm = ncm_scores.max(1)
                        correct_ncm += pred_ncm.eq(targets).sum().item()

                        # 3. Ensemble Prediction (Flat: Head Logits + Scaled NCM Cosine Scores)
                        ens_scores = head_scores + self.model.classifier.fc.scale * ncm_scores
                        _, pred_ens = ens_scores.max(1)
                        correct_ens += pred_ens.eq(targets).sum().item()
                        correct_ens_top5 += calc_topk_correct(ens_scores, targets, k=5)

                        # 4. Measure 3 Upgrade: Robust Bayesian Soft Multi-Task Mixture Routing
                        # Stage 1: Compute robust task affinity scores from Top-3 Memory Bank prototypes per task
                        task_affinities = []
                        top_k_proto = min(3, classes_per_task)
                        for k in range(num_tasks_seen):
                            k_start = k * classes_per_task
                            k_end = k_start + classes_per_task
                            # Mean of Top-3 prototypes in task k prevents single-prototype noise from skewing routing
                            task_k_topk = ncm_scores[:, k_start:k_end].topk(top_k_proto, dim=1)[0].mean(dim=1)
                            task_affinities.append(task_k_topk)
                        task_affinities = torch.stack(task_affinities, dim=1)  # [B, num_tasks_seen]
                        pred_task = task_affinities.argmax(dim=1)  # [B]
                        true_task = targets // classes_per_task
                        correct_task_routing += pred_task.eq(true_task).sum().item()

                        # Stage 2: Bayesian soft mixture across candidate tasks: P(c) = P(Task k) * P(c | Task k)
                        # Soft temperature T=0.25 prevents extreme zeroing of viable task hypotheses
                        task_weights = F.softmax(task_affinities / 0.25, dim=1)  # [B, num_tasks_seen]
                        hier_scores = torch.zeros(inputs.size(0), seen_classes, device=self.device)
                        for k in range(num_tasks_seen):
                            k_start = k * classes_per_task
                            k_end = k_start + classes_per_task
                            intra_probs = F.softmax(ens_scores[:, k_start:k_end], dim=1)  # [B, classes_per_task]
                            hier_scores[:, k_start:k_end] = task_weights[:, k:k+1] * intra_probs

                        _, pred_hier = hier_scores.max(1)
                        correct_hier += pred_hier.eq(targets).sum().item()
                        correct_hier_top5 += calc_topk_correct(hier_scores, targets, k=5)
                    else:
                        correct_ncm += correct_head
                        correct_ens += correct_head
                        correct_ens_top5 += correct_head_top5
                        correct_hier += correct_head
                        correct_hier_top5 += correct_head_top5
                        correct_task_routing += targets.size(0)

                head_acc = correct_head / total if total > 0 else 0.0
                head_top5 = correct_head_top5 / total if total > 0 else 0.0
                ncm_acc = correct_ncm / total if total > 0 else 0.0
                ens_acc = correct_ens / total if total > 0 else 0.0
                ens_top5 = correct_ens_top5 / total if total > 0 else 0.0
                hier_acc = correct_hier / total if total > 0 else 0.0
                hier_top5 = correct_hier_top5 / total if total > 0 else 0.0
                routing_acc = correct_task_routing / total if total > 0 else 0.0

                task_head_accuracies.append(head_acc)
                task_head_top5_accuracies.append(head_top5)
                task_ncm_accuracies.append(ncm_acc)
                task_ens_accuracies.append(ens_acc)
                task_ens_top5_accuracies.append(ens_top5)
                task_hier_accuracies.append(hier_acc)
                task_hier_top5_accuracies.append(hier_top5)
                task_routing_accuracies.append(routing_acc)

                print(f"  -> Task {t+1} Hierarchical: {hier_acc:.4f} (Top-5: {hier_top5:.4f}) | Ensemble: {ens_acc:.4f} (Top-5: {ens_top5:.4f}) | Head: {head_acc:.4f} (Top-5: {head_top5:.4f}) | NCM: {ncm_acc:.4f} [Task Routing: {routing_acc:.4f}]", flush=True)

        # Store matrices
        self.accuracy_matrix.append(task_hier_accuracies)
        self.accuracy_matrix_hier.append(task_hier_accuracies)
        self.accuracy_matrix_ens.append(task_ens_accuracies)
        self.accuracy_matrix_ncm.append(task_ncm_accuracies)
        self.accuracy_matrix_head.append(task_head_accuracies)
        self.accuracy_matrix_top5_ens.append(task_ens_top5_accuracies)
        self.accuracy_matrix_top5_head.append(task_head_top5_accuracies)
        self.accuracy_matrix_top5_hier.append(task_hier_top5_accuracies)

        # Save checkpoint (model weights + accuracy history + EWC state)
        ckpt_path = os.path.join(self.checkpoint_dir, f"mavis_task_{current_task_id+1}.pth")
        torch.save(self.model.state_dict(), ckpt_path)
        if self.ewc_module is not None:
            ewc_ckpt_path = os.path.join(self.checkpoint_dir, f"mavis_ewc_task_{current_task_id+1}.pth")
            torch.save(self.ewc_module.state_dict(), ewc_ckpt_path)
        metrics_ckpt = os.path.join(self.checkpoint_dir, "accuracy_matrix.json")
        metrics_hier_ckpt = os.path.join(self.checkpoint_dir, "accuracy_matrix_hier.json")
        metrics_ens_ckpt = os.path.join(self.checkpoint_dir, "accuracy_matrix_ens.json")
        metrics_ncm_ckpt = os.path.join(self.checkpoint_dir, "accuracy_matrix_ncm.json")
        metrics_head_ckpt = os.path.join(self.checkpoint_dir, "accuracy_matrix_head.json")
        metrics_top5_ens_ckpt = os.path.join(self.checkpoint_dir, "accuracy_matrix_top5_ens.json")
        metrics_top5_head_ckpt = os.path.join(self.checkpoint_dir, "accuracy_matrix_top5_head.json")
        metrics_top5_hier_ckpt = os.path.join(self.checkpoint_dir, "accuracy_matrix_top5_hier.json")
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
        with open(metrics_top5_ens_ckpt, "w") as f:
            json.dump(self.accuracy_matrix_top5_ens, f, indent=2)
        with open(metrics_top5_head_ckpt, "w") as f:
            json.dump(self.accuracy_matrix_top5_head, f, indent=2)
        with open(metrics_top5_hier_ckpt, "w") as f:
            json.dump(self.accuracy_matrix_top5_hier, f, indent=2)

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
