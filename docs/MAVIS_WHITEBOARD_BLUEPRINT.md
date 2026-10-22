# 🎨 MAVIS: Master Whiteboard Blueprint & Deep Architectural Compendium
### Memory Augmented Visual Incremental System for Class-Incremental Continual Learning
**Designed for Canva Architectural Whiteboard Representation**  
**Dataset:** Tiny ImageNet (200 Classes, 10 Incremental Tasks × 20 Classes/Task, 64×64×3)  
**Constraint:** Strict Class-Incremental Learning (CIL) from scratch — **Zero Pretraining Allowed**

---

## 🧭 Master Canvas Layout: 6 Spatial Zones

When laying out your Canva whiteboard, organize your canvas into **6 primary functional zones**:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                            CANVA WHITEBOARD LAYOUT                                              │
├───────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────┤
│ ZONE 1: INPUT PIPELINE & CNN BACKBONE     │ ZONE 2: MULTI-PROTOTYPE MEMORY BANK & ATTENTION                     │
│ • Raw Image Input: (B, 3, 64, 64)         │ • Memory Bank Buffer: [10, 20, K, 512]                              │
│ • 7-Stage Data Augmentation Pipeline      │ • Online Running Prototypes EMA Buffer: [20, 512]                   │
│ • Conv1 + GroupNorm (No MaxPool!)         │ • Scaled Dot-Product Multi-Head Attention (H=4, d_k=128)            │
│ • 4-Stage Wide ResNet-18 (SE-BasicBlocks) │ • Temperature Sharpening (tau=0.50) & Learned "No-Memory" Embed     │
│ • GroupNorm(32) & LayerNorm(512)          │ • 3D Tensor BMM Formulation (Bypassing Triton JIT)                  │
├───────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ ZONE 3: GATED FUSION & COSINE CLASSIFIER  │ ZONE 4: UNIFIED LOSS ENGINE & PARAMETER PROTECTION                  │
│ • Feature Concatenation: (B, 1024)        │ • Active Plasticity: Task-Masked L_CE + Margin (m=0.20) + L_proto   │
│ • Adaptive Gating: Linear + Sigmoid       │ • Representation Retention: L_distill (E_0 & E_t-1) + PODNet L_spat │
│ • Clamped Interpolation: g in [0.05, 0.95]│ • Semantic Separation: Top-K L_repel (K=32, rho=0.20) + L_recon     │
│ • CosineLinear Head: L2 Normalized W & f  │ • Exemplar Generation: Stratified Gaussian L_replay (N=256)         │
│ • Additive Cosine Margin: cos(theta) - m  │ • Parameter Curvature: Online EWC (lambda=5000, gamma=0.88)         │
├───────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ ZONE 5: POST-TASK TRANSITION & ALIGNMENT  │ ZONE 6: TEST-TIME INFERENCE & BAYESIAN ROUTING                      │
│ • Multi-Prototype K-Means Locking (K=10)  │ • Model Evaluation Mode (All 200 Classes, No Task ID)               │
│ • Dynamic Multi-Task Prototype Re-Align   │ • Predictor 1: Aligned Linear Head Softmax (tau=0.90)               │
│ • Weight Aligning (WA) Temperature Parity │ • Predictor 2: Multi-Prototype Nearest Class Mean (NCM, tau=0.05)   │
│ • Balanced Classifier Finetuning (AdamW)  │ • Predictor 3: Probability-Calibrated Dynamic Ensemble (alpha)      │
│ • Anchor Snapshots: Base E_0 & Step E_t-1 │ • Predictor 4: Zero-Parameter Bayesian Hierarchical Soft Mixture    │
│ • Online Fisher Accumulation (N=2000)     │   (Double-Softmax Mathematical Fix: P(c|Task k) = P_ens(c)/P_task)  │
└───────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────┘
```

---

# 📦 ZONE 1: Input Pipeline & Convolutional Feature Extractor (Backbone)

### 1. Input Tensor Specification
* **Dimensions:** $[B, C, H, W] = [B, 3, 64, 64]$
* **Batch Size ($B$):** $128$ (Training), $64$ (Classifier Finetuning & Chunked Inference)
* **Color Channels ($C$):** $3$ (RGB)
* **Spatial Resolution ($H \times W$):** $64 \times 64$ pixels

### 2. Data Augmentation Pipeline (Ordered Sequence)
1. **Random Crop with Reflection Padding:**
   * Padding: $4$ pixels $\to$ Canvas becomes $72 \times 72$
   * Crop: Centered or random crop back to $64 \times 64$
2. **Random Horizontal Flip:** Probability $p = 0.50$
3. **Random Rotation:** Angle $\theta \in [-10^\circ, +10^\circ]$
4. **Color Jitter:** Factor $0.20$ (Brightness $\pm 20\%$, Contrast $\pm 20\%$, Saturation $\pm 20\%$, Hue $\pm 10\%$)
5. **ToTensor Conversion:** Converts PIL Image to PyTorch Tensor $\in [0.0, 1.0]$
6. **Calibrated Random Erasing (Cutout):**
   * Probability $p = 0.25$
   * Area Scale: $(0.02, 0.20)$ of total image area
   * Aspect Ratio: $(0.3, 3.3)$
   * Value: $0$ (Black patch) — prevents shortcut feature learning
7. **Channel-wise Normalization (ImageNet Standard):**
   * Mean: $\mu = [0.485, 0.456, 0.406]$
   * Std: $\sigma = [0.229, 0.224, 0.225]$
   * Transformation: $x_{\text{norm}} = (x - \mu) / \sigma$

---

### 3. CNN Backbone Architecture: Wide ResNet-18 (64×64 Continual Variant)

> **CRITICAL ARCHITECTURAL DECISION:**  
> Standard ResNet-18 uses a $7 \times 7$ convolution with stride $2$ followed by a $3 \times 3$ MaxPool (stride $2$). On $64 \times 64$ images, this aggressively reduces spatial dimensions to $16 \times 16$ before the first residual block, destroying critical fine-grained features.  
> **MAVIS Fix:** Uses a single $3 \times 3$ convolution with stride $1$ and **ZERO MaxPool layers**.

#### Initial Convolutional Layer (Stem):
* **Operation:** `nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)`
* **Normalization:** `nn.GroupNorm(num_groups=32, num_channels=128)`
* **Activation:** `nn.ReLU(inplace=True)`
* **Output Tensor Dimension:** $[B, 128, 64, 64]$

> **WHY GROUPNORM INSTEAD OF BATCHNORM?**  
> In Class-Incremental Learning, BatchNorm maintains running estimates of mean $\mu$ and variance $\sigma^2$. When training Task $t$, incoming batches contain *only* Task $t$ images. The running statistics rapidly drift toward Task $t$, severely corrupting feature extraction for historical tasks ($1 \dots t-1$). GroupNorm computes normalization statistics over groups of 32 channels independently per sample, completely eliminating inter-task statistic contamination!

---

#### The Residual Building Block: `BasicBlock` with Squeeze-and-Excitation (`SEBlock`)
Each residual block processes features through two convolutional layers and a channel-attention recalibration block:

```
                      Input x: [B, C_in, H, W]
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
   Conv1 (3x3, stride=s)             Shortcut Path
   GroupNorm(32, C_out)              (Identity if s=1 & C_in=C_out;
   ReLU                              else 1x1 Conv + GroupNorm)
   Dropout2d(p=0.5 if enabled)            │
   Conv2 (3x3, stride=1)                  │
   GroupNorm(32, C_out)                   │
        │                                 │
        ▼                                 │
   SEBlock (Channel Attention)            │
   • AdaptiveAvgPool2d(1) -> [B, C, 1, 1] │
   • Linear(C -> C//16) + ReLU            │
   • Linear(C//16 -> C) + Sigmoid         │
   • Multiply: x * w_c                    │
        │                                 │
        └────────────────┬────────────────┘
                         ▼
                   Elementwise Add (+)
                         ▼
                       ReLU
                         ▼
                   Output: [B, C_out, H', W']
```

#### Layer-by-Layer Dimension Flow:
1. **Stem:** $3 \to 128$ channels. Output: $[B, 128, 64, 64]$
2. **Layer 1 (2 BasicBlocks):**
   * Block 1: `in_planes=128, planes=128, stride=1`. Output: $[B, 128, 64, 64]$
   * Block 2: `in_planes=128, planes=128, stride=1`. Output: $[B, 128, 64, 64]$
3. **Layer 2 (2 BasicBlocks):**
   * Block 1: `in_planes=128, planes=256, stride=2` (Shortcut: $1 \times 1$ conv, stride 2). Output: $[B, 256, 32, 32]$
   * Block 2: `in_planes=256, planes=256, stride=1`. Output: $[B, 256, 32, 32]$
4. **Layer 3 (2 BasicBlocks) — Spatial Distillation Tap 1 ($f_3$):**
   * Block 1: `in_planes=256, planes=512, stride=2` (Shortcut: $1 \times 1$ conv, stride 2). Output: $[B, 512, 16, 16]$
   * Block 2: `in_planes=512, planes=512, stride=1`. Output: $[B, 512, 16, 16]$
   * **Spatial Tap $f_3$ Shape:** $[B, 512, 16, 16]$
5. **Layer 4 (2 BasicBlocks) — Spatial Distillation Tap 2 ($f_4$):**
   * Block 1: `in_planes=512, planes=1024, stride=2` (Shortcut: $1 \times 1$ conv, stride 2). Output: $[B, 1024, 8, 8]$
   * Block 2: `in_planes=1024, planes=1024, stride=1`. Output: $[B, 1024, 8, 8]$
   * **Spatial Tap $f_4$ Shape:** $[B, 1024, 8, 8]$

#### Global Feature Pooling & Latent Projection:
* **Adaptive Average Pooling:** `nn.AdaptiveAvgPool2d((1, 1))` $\to [B, 1024, 1, 1]$
* **Flatten:** $\to [B, 1024]$
* **Linear Projection:** `nn.Linear(in_features=1024, out_features=512, bias=True)` $\to [B, 512]$
* **Layer Normalization:** `nn.LayerNorm(512)` $\to$ Produces unit-variance latent feature $f_c \in \mathbb{R}^{B \times 512}$

---

# 🧠 ZONE 2: Multi-Prototype Neural Memory Bank & Scaled Attention Retrieval

### 1. Memory Bank Tensor Data Structure (`MemoryBank`)
The memory bank is implemented entirely with registered persistent buffers (`register_buffer`), meaning it is serialized in `state_dict` but never touched by the optimizer:
* **Primary Memory Tensor (`memory`):**
  * Shape: $[T, C_{\text{task}}, K, D] = [10, 20, 10, 512]$
  * Total Stored Centroids: $10 \times 20 \times 10 = 2,000$ prototype vectors.
  * Memory Footprint: $2000 \times 512 \times 4 \text{ bytes} \approx 4.096 \text{ MB}$ (Extremely compact!).
* **Class Variances Tensor (`class_variances`):**
  * Shape: $[T, C_{\text{task}}, D] = [10, 20, 512]$
  * Stores empirical feature variances $\sigma^2_{t, c} \in \mathbb{R}^{512}$ along each latent dimension for Gaussian replay.
* **Online Running Prototypes Buffer (`running_prototypes`):**
  * Shape: $[C_{\text{task}}, D] = [20, 512]$
  * Stores online Exponential Moving Average (EMA) centroids for the active task during training:
    $$\mu_{c}^{(k)} = 0.92 \cdot \mu_{c}^{(k-1)} + 0.08 \cdot \bar{f}_{\text{batch}, c}$$
* **Population Mask (`is_populated`):**
  * Shape: $[10]$ (boolean flags indicating which task slots are locked).

---

### 2. Multi-Prototype Clustering: Pure PyTorch K-Means
Rather than saving a single class mean, MAVIS clusters the representations of class $c$ into $K=10$ sub-cluster centroids to capture multimodal visual modes (lighting, camera angles, sub-breeds):
1. Compute all feature representations for class $c$: $X_c = \{f_i \in \mathbb{R}^{512}\}_{i=1}^{N_c}$.
2. Initialize centroids using K-Means++ style spread spacing along features:
   $$\mu_k^{(0)} = X_c\left[\text{round}\left(i \cdot \frac{N_c - 1}{K - 1}\right)\right], \quad k=0 \dots K-1$$
3. Run 10 iterations of Euclidean assignment and update:
   $$z_i = \arg\min_{k} \|f_i - \mu_k\|_2^2$$
   $$\mu_k \leftarrow \frac{1}{|S_k|} \sum_{i \in S_k} f_i$$
4. L2-normalize centroids to lie on the unit hypersphere:
   $$\mu_k \leftarrow \frac{\mu_k}{\|\mu_k\|_2}$$

---

### 3. Scaled Dot-Product Multi-Head Attention (`MemoryAttention`)

```
 Current Feature f_c: [B, 512]             Memory Bank Prototypes M: [N_past, 512]
          │                                                  │
          ▼                                                  ▼
     Unsqueeze(1)                                    Unsqueeze(0).repeat(B)
     Query: [B, 1, 512]                              Key/Value: [B, N_past, 512]
          │                                                  │
   Linear Proj (W_Q)                                  Linear Proj (W_K, W_V)
          │                                                  │
          ├──────────────────────────┬───────────────────────┘
          ▼                          ▼
       Q: [B*H, 1, d_k]           K: [B*H, N_past, d_k]
          │                          │
          └───────────┬──────────────┘
                      ▼
            torch.bmm(Q, K.transpose(1, 2))
                      ▼
           Raw Scores: [B*H, 1, N_past]
                      ▼
         Scale: Scores / (sqrt(d_k) * tau)   [tau = 0.50]
                      ▼
             Softmax(dim=-1)
                      ▼
             Dropout(p=0.10)
                      ▼
          Attention Weights: [B*H, 1, N_past]
                      │
                      ├──────────────────────────┐
                      ▼                          ▼
          torch.bmm(Attn, V)              V: [B*H, N_past, d_k]
                      ▼
              Out: [B*H, 1, d_k]
                      ▼
              Reshape: [B, 1, 512]
                      ▼
              Linear Proj (W_out)
                      ▼
               Squeeze + LayerNorm
                      ▼
          Retrieved Feature f_m: [B, 512]
```

#### Exact Mathematical & Dimensional Steps:
* **Dimensions:** Feature dimension $D = 512$, Number of Heads $H = 4$, Head Dimension $d_k = D / H = 128$.
* **Projection Matrices:**
  * $W_Q \in \mathbb{R}^{512 \times 512}$
  * $W_K \in \mathbb{R}^{512 \times 512}$
  * $W_V \in \mathbb{R}^{512 \times 512}$
  * $W_{\text{out}} \in \mathbb{R}^{512 \times 512}$
* **Task 1 Handling (Learned No-Memory Embedding):**
  When $t=0$ (no historical prototypes exist), attention returns a learned parameter vector:
  $$\text{no\_memory\_embed} \sim \mathcal{N}(0, 0.01^2) \in \mathbb{R}^{1 \times 512}$$
  expanded to $[B, 512]$. This avoids the catastrophic distribution shift of returning raw zeros!
* **HPC Pure PyTorch Kernel (Bypassing Triton C++ JIT):**
  4D tensor multiplications `torch.matmul([B, H, 1, d], [B, H, d, N])` trigger JIT Triton kernel compilation that fails on restricted HPC environments. MAVIS reshapes tensors into 3D:
  $$Q_{\text{3D}} = Q.\text{reshape}(B \cdot H, 1, d_k), \quad K_{\text{3D}} = K.\text{reshape}(B \cdot H, N, d_k)$$
  and invokes pre-compiled cuBLAS via `torch.bmm`.
* **Temperature Sharpening:**
  $$\text{Scores}_{\text{3D}} = \frac{Q_{\text{3D}} \cdot K_{\text{3D}}^T}{\sqrt{d_k} \cdot \tau_{\text{attn}}} = \frac{Q_{\text{3D}} \cdot K_{\text{3D}}^T}{\sqrt{128} \cdot 0.50}$$
  Sharpening with $\tau = 0.50$ prevents attention probabilities from dispersing diffusely across 1,800+ memory keys.

---

# ⚖️ ZONE 3: Gated Feature Fusion & Balanced Cosine Linear Classifier

### 1. Dynamic Gated Fusion Block
Combines the live visual feature $f_c$ with the memory-retrieved feature $f_m$:
1. **Concatenate Features:**
   $$z = [f_c \,;\, f_m] \in \mathbb{R}^{B \times 1024}$$
2. **Linear Projection & Sigmoid Gating:**
   $$\text{raw\_gate} = \sigma\left( W_g z + b_g \right) \in (0, 1)^{B \times 512}$$
   where $W_g \in \mathbb{R}^{512 \times 1024}$ and $b_g \in \mathbb{R}^{512}$.
   *Initialization:* $W_g = \mathbf{0}, b_g = \mathbf{0}$, ensuring initial gate is exactly $0.50$ (balanced 50/50 prior).
3. **Clamped Affine Dynamic Scaling:**
   To guarantee that neither the live encoder nor the memory retrieval is completely starved:
   $$g = 0.05 + 0.90 \cdot \text{raw\_gate} \in [0.05, 0.95]$$
4. **Fused Feature Synthesis:**
   $$f_{\text{fused}} = g \odot f_c + (1.0 - g) \odot f_m \in \mathbb{R}^{B \times 512}$$

---

### 2. Balanced Cosine Linear Head (`CosineLinear`)

> **WHY COSINE CLASSIFIER INSTEAD OF STANDARD LINEAR?**  
> In standard linear classifiers ($y = W x + b$), weights for newly trained classes receive frequent gradient updates, growing large $L_2$ norms ($\|W_{\text{new}}\| \gg \|W_{\text{old}}\|$). During test time, the softmax denominator is dominated by new classes, causing severe recency bias (>85% false predictions).  
> **CosineLinear** normalizes both weights and features to the unit hypersphere $\mathbb{S}^{511}$, ensuring logits represent purely angular cosine similarities!

* **Weight Parameter Matrix:** $W \in \mathbb{R}^{200 \times 512}$ (No bias).
* **Buffer Vector:** $\text{task\_scale} \in \mathbb{R}^{200}$ (Initialized to $1.0$).
* **L2-Normalization:**
  $$\tilde{f} = \frac{f_{\text{fused}}}{\|f_{\text{fused}}\|_2}, \quad \tilde{W}_j = \frac{W_j}{\|W_j\|_2}, \quad \forall j \in \{0 \dots 199\}$$
* **Cosine Matrix Multiplication:**
  $$\text{CosineSim}_{i, j} = \tilde{f}_i \cdot \tilde{W}_j \in [-1.0, +1.0]$$
* **Additive Cosine Margin ($m = 0.20$):**
  During training on ground-truth targets $y_i$, subtract margin $m$ from the target angle cosine:
  $$\text{MarginSim}_{i, j} = \text{CosineSim}_{i, j} - m \cdot \mathbb{I}[j = y_i]$$
* **Temperature Scaling:**
  $$\text{Logits}_{i, j} = s \cdot \text{MarginSim}_{i, j} \cdot \text{task\_scale}_j$$
  where $s = 24.0$. Output dimension: $[B, 200]$.

---

# 📉 ZONE 4: Unified Continual Loss Engine (All 9 Losses)

During incremental training on Task $t \ge 1$, the model optimizes the unified objective:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \alpha_{\text{warmup}} \left( \lambda_{\text{distill}} \mathcal{L}_{\text{distill}} + \lambda_{\text{spatial}} \mathcal{L}_{\text{spatial}} \right) + \lambda_{\text{proto}} \mathcal{L}_{\text{proto}} + \lambda_{\text{repel}} \mathcal{L}_{\text{repel}} + 1.0 \cdot \mathcal{L}_{\text{recon}} + \lambda_{\text{replay}} \mathcal{L}_{\text{replay}} + \mathcal{L}_{\text{EWC}}$$

---

### Component-by-Component Mathematical Breakdown

| Loss Name | Mathematical Formula | Variables & Shapes | Weight / Hyperparams | Purpose / What It Prevents |
|---|---|---|:---:|---|
| **1. Task-Masked Active CE** ($\mathcal{L}_{\text{CE}}$) | $\mathcal{L}_{\text{CE}} = - \frac{1}{B} \sum_{i=1}^B \sum_{c=\text{start}}^{\text{end}} y_{i, c} \log \frac{e^{z_{i, c}}}{\sum_{j} e^{z_{i, j}}}$ | $z \in \mathbb{R}^{B \times 20}$, $y \in \{0 \dots 19\}$ | Label Smooth $\epsilon=0.02$, Margin $m=0.20$ | Drives intra-task learning. Masking active 20 neurons prevents destructive positive gradients to historical weights! |
| **2. Multi-Anchor Distillation** ($\mathcal{L}_{\text{distill}}$) | $0.5 \left(1 - \cos(f_c, E_{t-1}(x))\right) + 0.5 \left(1 - \cos(f_c, E_0(x))\right)$ | $f_c, E_{t-1}(x), E_0(x) \in \mathbb{R}^{B \times 512}$ | $\lambda_{\text{distill}} = 3.0$, Warmup $\alpha \in [0.5, 1.0]$ | Prevents latent coordinate rotation by anchoring to both preceding task ($E_{t-1}$) and immutable Task 1 base ($E_0$). |
| **3. Multi-Axis PODNet Spatial** ($\mathcal{L}_{\text{spatial}}$) | $\frac{1}{2L} \sum_{l \in \{3, 4\}} \left( \|\bar{S}_w^{\text{curr}} - \bar{S}_w^{\text{old}}\|_2^2 + \|\bar{S}_h^{\text{curr}} - \bar{S}_h^{\text{old}}\|_2^2 \right)$ | $f_3 \in \mathbb{R}^{B \times 512 \times 16 \times 16}$, $f_4 \in \mathbb{R}^{B \times 1024 \times 8 \times 8}$ | $\lambda_{\text{spatial}} = 1.5$ | Preserves intermediate spatial pooling representations (width & height slices) of texture, edges, and shape hierarchies. |
| **4. InfoNCE Prototype Alignment** ($\mathcal{L}_{\text{proto}}$) | $- \frac{1}{B} \sum_{i=1}^B \log \frac{\exp(\tilde{f}_i \cdot \tilde{\mu}_{y_i} / \tau_{\text{proto}})}{\sum_{c=1}^{20} \exp(\tilde{f}_i \cdot \tilde{\mu}_c / \tau_{\text{proto}})}$ | $\tilde{f} \in \mathbb{R}^{B \times 512}$, $\tilde{\mu} \in \mathbb{R}^{20 \times 512}$ | $\lambda_{\text{proto}} = 0.5$, Temperature $\tau = 0.20$ | Pulls live features into tight clusters around running active centroids while repelling other active class centroids. |
| **5. Top-K Hard Negative Repulsion** ($\mathcal{L}_{\text{repel}}$) | $\frac{1}{B \cdot K_{\text{neg}}} \sum_{i=1}^B \sum_{m \in \text{Top}32} \max(0, \cos(f_i, m) - \rho)^2$ | $m \in \mathcal{M}_{\text{past}}$, $K_{\text{neg}} = \min(32, |\mathcal{M}|)$ | $\lambda_{\text{repel}} = 2.0$, Threshold $\rho = 0.20$ | Pushes new task features away from the 32 closest historical prototypes. Eliminates diffuse gradient cancellation! |
| **6. Attention Self-Reconstruction** ($\mathcal{L}_{\text{recon}}$) | $\frac{1}{N_{\text{samp}}} \sum_{m} \|\text{Attention}(m, \mathcal{M}_{\text{past}}) - m\|_2^2$ | $m \in \mathbb{R}^{128 \times 512}$, $\mathcal{M}_{\text{past}} \in \mathbb{R}^{N \times 512}$ | Weight $= 1.0$, Subsample $= 128$ | Guarantees that query-key projections ($W_Q, W_K, W_V$) maintain high-fidelity associative retrieval of old memories. |
| **7. Stratified Gaussian Replay** ($\mathcal{L}_{\text{replay}}$) | $- \frac{1}{N_{\text{sub}}} \sum_{i} \log \frac{e^{s \cdot \cos(\text{fuse}(\tilde{x}_i), W_{y_i})}}{\sum_{j} e^{s \cdot \cos(\text{fuse}(\tilde{x}_i), W_j)}}$ | $\tilde{x} = \mu + 0.5 \sigma \epsilon$, Stratified budget $= 256$ | $\lambda_{\text{replay}} = 2.0$, $O(1)$ constant budget | Provides synthetic feature clouds along true empirical covariance ellipsoids. Replays past classes during live training. |
| **8. Online EWC Regularization** ($\mathcal{L}_{\text{EWC}}$) | $\frac{\lambda_{\text{EWC}}}{2} \sum_{p \in \Theta} F_{\text{running}, p} \cdot (\theta_p - \theta_p^*)^2$ | $F \in \mathbb{R}^{|\Theta|}$, $\theta^* \in \mathbb{R}^{|\Theta|}$ | $\lambda_{\text{EWC}} = 5000.0$, Decay $\gamma = 0.88$ | Parameter curvature penalty. Penalizes updates to CNN weights along high-information Fisher directions. |
| **9. Post-Task Balanced Finetune** ($\mathcal{L}_{\text{finetune}}$) | $\text{CrossEntropy}(\text{Classifier}(\text{fuse}(\mu_c + \text{noise})), y_c)$ | Frozen Encoder, $C_{\text{seen}} = 20 \times (t+1)$ | 25 Epochs AdamW, LR $= 0.005$, Draws $= 10$ | Post-task classifier re-calibration across all seen classes. Eliminates classifier recency bias. |

---

# 🔄 ZONE 5: Post-Task Transition & Alignment Pipeline

When training completes on Task $t$, MAVIS executes a 6-step transition protocol before moving to Task $t+1$:

```
                       Task t Training Completed (50 Epochs)
                                        │
                                        ▼
             ┌──────────────────────────────────────────────────────┐
             │ Step 1: Memory Bank Locking via K-Means              │
             │ Extract features of all Task t images; compute K=10  │
             │ sub-centroids & diagonal variances; lock in memory.  │
             └──────────────────────────┬───────────────────────────┘
                                        │
                                        ▼
             ┌──────────────────────────────────────────────────────┐
             │ Step 2: Dynamic Multi-Task Prototype Re-Alignment    │
             │ Loop over ALL historical tasks 0..t-1 and re-encode  │
             │ their prototypes through current encoder E_t.        │
             │ Completely solves Representation Drift!              │
             └──────────────────────────┬───────────────────────────┘
                                        │
                                        ▼
             ┌──────────────────────────────────────────────────────┐
             │ Step 3: Weight Aligning (WA) Temperature Parity      │
             │ Equalize task_scale to 1.0 across all seen classes:  │
             │ task_scale[:(t+1)*20] = 1.0                          │
             └──────────────────────────┬───────────────────────────┘
                                        │
                                        ▼
             ┌──────────────────────────────────────────────────────┐
             │ Step 4: Post-Task Balanced Classifier Finetuning     │
             │ Freeze CNN backbone & attention. Train classifier on │
             │ Gaussian-augmented prototypes across all seen classes│
             │ for 25 epochs using AdamW (targets=None).            │
             └──────────────────────────┬───────────────────────────┘
                                        │
                                        ▼
             ┌──────────────────────────────────────────────────────┐
             │ Step 5: Anchor Snapshots                             │
             │ - If Task 0: Snapshot Base Anchor E_0 (immutable).   │
             │ - Always: Snapshot Step Anchor E_t <- E_curr.        │
             └──────────────────────────┬───────────────────────────┘
                                        │
                                        ▼
             ┌──────────────────────────────────────────────────────┐
             │ Step 6: Online EWC Fisher Matrix Update              │
             │ Compute empirical Fisher diagonal over 2000 samples: │
             │ F_new = E[(d log p / d theta)^2]                     │
             │ Accumulate: F_running = gamma * F_running + F_new    │
             │ Snapshot parameter anchor: theta* <- theta           │
             └──────────────────────────────────────────────────────┘
```

---

# 🎯 ZONE 6: Test-Time Inference & Zero-Parameter Bayesian Routing

At evaluation time, a test image $x \in \mathbb{R}^{B \times 3 \times 64 \times 64}$ arrives. **No task identifier is provided.** The model must predict among all $C_{\text{seen}} = 20 \times (t+1)$ classes.

MAVIS processes the input through **4 distinct predictors**:

```
                       Test Image Input: [B, 3, 64, 64]
                                      │
                                      ▼
                             CNN Encoder E_curr
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
     Current Feature f_c: [B, 512]               Intermediate Spatial Maps
               │
               ├─────────────────────────────────────────────┐
               ▼                                             ▼
     Multi-Head Attention                         Multi-Prototype NCM
   (Retrieves from all populated memories)      sim_c = max_{k} cos(f_c, mu_{c, k})
               │                                             │
               ▼                                             ▼
      Retrieved Feature f_m                       NCM Scores: [B, C_seen]
               │                                             │
               ▼                                             │
      Dynamic Gated Fusion                                   │
      f_fused = g*f_c + (1-g)*f_m                            │
               │                                             │
               ▼                                             │
      Cosine Linear Head                                     │
               │                                             │
               ▼                                             │
      Head Scores: [B, C_seen]                               │
               │                                             │
               ├──────────────────────┬──────────────────────┤
               ▼                      ▼                      ▼
        PREDICTOR 1:            PREDICTOR 2:          PREDICTOR 3:
     Aligned Linear Head     Multi-Prototype NCM   Dynamic Ensemble
     P_head = Softmax(z/0.9) P_ncm = Softmax(s/0.05) P_ens = (1-a)P_head + a*P_ncm
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      │
                                      ▼
                                 PREDICTOR 4:
                Zero-Parameter Bayesian Hierarchical Mixture
```

---

### The Mathematical Formulations of the 4 Predictors

#### Predictor 1: Aligned Linear Head
$$P_{\text{head}}(c \mid x) = \text{Softmax}\left(\frac{\text{Logits}[:, :C_{\text{seen}}]}{0.90}\right)_c$$

#### Predictor 2: Multi-Prototype Nearest Class Mean (NCM)
For class $c$ with $K$ prototypes $\{\mu_{c, 1} \dots \mu_{c, K}\}$:
$$\mathcal{S}_{\text{ncm}}(c) = \max_{k \in \{1 \dots K\}} \left( \frac{f_c \cdot \mu_{c, k}}{\|f_c\|_2 \|\mu_{c, k}\|_2} \right)$$
$$P_{\text{ncm}}(c \mid x) = \text{Softmax}\left(\frac{\mathcal{S}_{\text{ncm}}}{0.05}\right)_c$$

#### Predictor 3: Probability-Calibrated Dynamic Ensemble
Eliminates score scale disparity between NCM and Linear Head using a task-age transition sigmoid:
$$\Delta t_c = t_{\text{current}} - \text{task}(c)$$
$$\alpha_c = \sigma\left(\frac{\Delta t_c - 1.0}{0.70}\right)$$
$$P_{\text{ens}}(c \mid x) = (1.0 - \alpha_c) \cdot P_{\text{head}}(c \mid x) + \alpha_c \cdot P_{\text{ncm}}(c \mid x)$$
*Behavior:* For current task ($\Delta t = 0$), $\alpha \approx 0.19$ (relies on Head). For older tasks ($\Delta t \ge 2$), $\alpha \ge 0.81$ (relies on frozen NCM memory).

#### Predictor 4: Zero-Parameter Bayesian Hierarchical Routing (Double-Softmax Fixed)
1. **Marginal Task Probability (Law of Total Probability):**
   $$P_{\text{marginal}}(\text{Task } k \mid x) = \sum_{c \in \text{Task } k} P_{\text{ens}}(c \mid x)$$
2. **Top-3 Prototype Geometric Affinity:**
   $$\mathcal{A}_k = \frac{1}{3} \sum_{m \in \text{Top3}(\text{Task } k)} \cos(f_c, m)$$
   $$P_{\text{aff}}(\text{Task } k \mid x) = \text{Softmax}\left(\frac{\mathcal{A}_k}{0.18}\right)$$
3. **Balanced Task Weight:**
   $$w_k = 0.50 \cdot P_{\text{marginal}}(\text{Task } k \mid x) + 0.50 \cdot P_{\text{aff}}(\text{Task } k \mid x)$$
4. **Conditional Class Probability (Bayes' Rule, Double-Softmax Mathematical Fix):**
   $$P(c \mid \text{Task } k) = \frac{P_{\text{ens}}(c \mid x)}{\sum_{c' \in \text{Task } k} P_{\text{ens}}(c' \mid x)}$$
5. **Hierarchical Class Score:**
   $$\mathcal{S}_{\text{hier}}(c) = w_k \cdot P(c \mid \text{Task } k)$$

---

# 🎨 Canva Whiteboard Construction Checklist

When drawing your board, follow this color-coding and block checklist:

| Block / Wire | Canva Shape | Recommended Color | Text / Mathematical Annotation |
|---|---|---|---|
| **Input Image** | Rounded Rectangle | `#4A90E2` (Soft Blue) | `Input x: (B, 3, 64, 64)` |
| **Augmentation Pipeline** | Dashed Container | `#E8F4FD` (Light Blue) | `Crop + Flip + Rotate(10°) + Jitter + Erasing(0.25)` |
| **Conv Stem** | Rectangle | `#F5A623` (Amber) | `Conv2D(3->128, 3x3) + GroupNorm(32) + ReLU` |
| **ResNet Layers 1-4** | Stack of 4 Rectangles | `#7ED321` (Green) | `L1: 128 (64x64) -> L2: 256 (32x32) -> L3: 512 (16x16) -> L4: 1024 (8x8)` |
| **SEBlock** | Diamond / Hexagon | `#BD10E0` (Purple) | `AvgPool -> Linear(C/16) -> ReLU -> Linear(C) -> Sigmoid` |
| **Pooling & Linear** | Rectangle | `#50E3C2` (Teal) | `AdaptiveAvgPool(1x1) -> Linear(1024->512) -> LayerNorm(512)` |
| **Current Feature $f_c$** | Circle / Pill | `#4A90E2` (Blue) | `$f_c \in \mathbb{R}^{B \times 512}$` |
| **Memory Bank** | Cylinder / Grid | `#9013FE` (Deep Violet) | `Buffer: [10 tasks, 20 cls, K=10 protos, 512-D]` |
| **Multi-Head Attention** | Large Rounded Box | `#B8E986` (Lime) | `Q=W_Q f_c, K=W_K M, V=W_V M, Softmax(QK^T / (tau*sqrt(d))) V` |
| **Retrieved Feature $f_m$** | Circle / Pill | `#9013FE` (Violet) | `$f_m \in \mathbb{R}^{B \times 512}$` |
| **Gated Fusion** | Circle with $\otimes$ and $\oplus$ | `#D0021B` (Coral Red) | `g = 0.05 + 0.90 * sigmoid(W_g [f_c; f_m]); fused = g*f_c + (1-g)*f_m` |
| **Cosine Classifier** | Rectangle | `#F8E71C` (Yellow) | `CosineLinear: scale=24.0, margin=0.20, norm(W) @ norm(f)` |
| **Unified Logits** | Output Pill | `#E67E22` (Orange) | `Logits: (B, 200)` |
| **EWC Regularizer** | Shield / Lock Symbol | `#D35400` (Rust Orange) | `L_EWC = (lambda/2) * sum(F_i * (theta_i - theta_i*)^2)` |
| **Distillation Anchors** | Two Frozen Boxes | `#95A5A6` (Slate Gray) | `Base Anchor E_0 (Task 1) + Step Anchor E_{t-1}` |
| **Bayesian Router** | Tree / Split Node | `#8E44AD` (Dark Purple) | `w_k * P_ens(c) / sum P_ens(c')` |
