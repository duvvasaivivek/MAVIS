"""
MAVIS Configuration Module
===========================
Central configuration for all hyperparameters and paths.
Initial values from spec §47 — these are starting values, NOT final optimized values.
"""

import os
import yaml
from dataclasses import dataclass, field, asdict, fields


# ─── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")


# ─── Configuration Dataclasses ───────────────────────────────────────────────

@dataclass
class DatasetConfig:
    """Dataset and continual-task configuration."""
    name: str = "tiny_imagenet"
    data_dir: str = os.path.join(DATA_DIR, "tiny-imagenet-200")
    image_size: int = 64
    num_channels: int = 3
    num_classes: int = 200
    classes_per_task: int = 20
    num_tasks: int = 10
    seed: int = 42


@dataclass
class AugmentationConfig:
    """Data augmentation settings."""
    random_flip_horizontal: bool = True
    random_crop_padding: int = 4
    color_jitter: bool = True
    color_jitter_factor: float = 0.2
    # Enhanced visual invariance (active for maximum generalization)
    random_rotation: bool = True
    random_rotation_degrees: int = 10
    random_erasing: bool = True
    random_erasing_prob: float = 0.25
    mixup: bool = False
    cutmix: bool = False


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 128
    epochs_per_task: int = 50  # 50-epoch optimal schedule
    learning_rate: float = 0.008
    lr_backbone_factor: float = 0.20   # Rebalanced backbone plasticity: 0.20x for tasks t >= 1 (was 0.10)
    label_smoothing: float = 0.02      # Lower loss floor to enable sharp convergence (>0.90+ train acc)
    optimizer: str = "adam"
    mixed_precision: bool = True
    seed: int = 42
    # Continual Learning loss weights (Plasticity Rebalance: moderated distillation)
    distill_lambda: float = 3.0         # Moderated Feature Distillation to E_0 and E_{t-1} (was 5.0)
    spatial_distill_lambda: float = 1.5 # Moderated Multi-Axis PODNet spatial distillation (was 3.0)
    proto_lambda: float = 0.5           # Relaxed centroid alignment to preserve sample variance
    repel_lambda: float = 2.0
    proto_replay_lambda: float = 2.0    # Strengthened Gaussian Covariance Prototype Replay
    proto_finetune_epochs: int = 25     # Deeper balanced classifier finetuning (expanded from 5)
    proto_finetune_draws: int = 5       # Calibrated Gaussian feature cloud (5 draws per prototype)
    # Progressive training stages
    smoke_test_epochs: int = 2
    smoke_test_tasks: int = 2


@dataclass
class EncoderConfig:
    """CNN encoder architecture."""
    feature_dim: int = 512
    # Channel counts per block — reduce if GPU OOM
    channels: list = field(default_factory=lambda: [64, 128, 256, 512])
    use_batch_norm: bool = True
    dropout_rate: float = 0.5


@dataclass
class ClassifierConfig:
    """Classifier head."""
    num_classes: int = 200
    margin: float = 0.20  # Additive cosine margin to enforce inter-class angular separation


@dataclass
class EWCConfig:
    """
    Elastic Weight Consolidation — LOSS-LEVEL regularization.
    
    EWC adds a penalty to the loss function during backpropagation:
        L_total = L_CE + L_distill + ... + (lambda/2) * Σ F_i * (θ_i − θ_i*)²
    
    It does NOT appear in the forward pass / data flow.
    Fisher values and old parameters are frozen reference tensors.
    """
    enabled: bool = True
    ewc_lambda: float = 5000.0  # Phase 5 calibrated: in equilibrium with distillation and CE
    fisher_samples: int = 2000
    gamma: float = 0.88         # Online EWC decay factor for older task Fisher contributions (calibrated for late-task plasticity)
    # Lambda sweep values for experiments
    lambda_sweep: list = field(default_factory=lambda: [500, 1000, 2000, 5000])


@dataclass
class MemoryConfig:
    """
    External Neural Memory — stores compact multi-prototype representations.
    
    Each class stores K=10 prototype vectors and covariance diagonals.
    Total capacity = 200 classes × 10 prototypes = 2000 entries.
    """
    capacity: int = 2000
    feature_dim: int = 512
    prototypes_per_class: int = 10  # Expanded K=10 sub-clusters per class for 43% higher visual coverage


@dataclass
class AttentionConfig:
    """
    Multi-Head Attention for memory retrieval.
    
    Operates in the forward pass:
        Q = W_Q · f_current
        K = W_K · M_memory
        V = W_V · M_memory
        Retrieved = softmax(QK^T / (τ · √d_k)) · V
    """
    num_heads: int = 4
    key_dim: int = 256
    query_dim: int = 256
    value_dim: int = 512
    temperature: float = 0.5  # Temperature-sharpened retrieval to eliminate diluted averaging
    # Heads sweep for experiments
    heads_sweep: list = field(default_factory=lambda: [1, 2, 4, 8])


@dataclass
class FusionConfig:
    """
    Gated Fusion — combines current and retrieved features.
    
    z = [f_current ; f_retrieved]
    g = sigmoid(W_g · z + b_g)
    f_final = g ⊙ f_current + (1-g) ⊙ f_retrieved
    """
    fusion_dim: int = 512


@dataclass
class MAVISConfig:
    """Complete MAVIS configuration — assembles all sub-configs."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    ewc: EWCConfig = field(default_factory=EWCConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    def save(self, path: str) -> None:
        """Save configuration to YAML."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
        print(f"[Config] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MAVISConfig":
        """Load configuration from YAML."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        config = cls()
        for section_name, section_data in data.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                section_cls = type(getattr(config, section_name))
                valid_fields = {f.name for f in fields(section_cls)}
                filtered_data = {k: v for k, v in section_data.items() if k in valid_fields}
                setattr(config, section_name, section_cls(**filtered_data))

        # Fall back to project data directory if serialized path does not exist on current OS
        if not os.path.exists(config.dataset.data_dir):
            default_data_dir = os.path.join(DATA_DIR, "tiny-imagenet-200")
            if os.path.exists(default_data_dir):
                config.dataset.data_dir = default_data_dir

        print(f"[Config] Loaded from {path}")
        return config

    def summary(self) -> str:
        """Print a human-readable summary."""
        lines = [
            "=" * 60,
            "MAVIS Configuration Summary",
            "=" * 60,
            f"  Dataset:       {self.dataset.name} ({self.dataset.num_classes} classes)",
            f"  Tasks:         {self.dataset.num_tasks} tasks × {self.dataset.classes_per_task} classes",
            f"  Image size:    {self.dataset.image_size}×{self.dataset.image_size}×{self.dataset.num_channels}",
            f"  Batch size:    {self.training.batch_size}",
            f"  Epochs/task:   {self.training.epochs_per_task}",
            f"  Learning rate: {self.training.learning_rate}",
            f"  Feature dim:   {self.encoder.feature_dim}",
            f"  EWC:           {'ON (lambda=' + str(self.ewc.ewc_lambda) + ')' if self.ewc.enabled else 'OFF'}",
            f"  Memory:        {self.memory.capacity} entries ({self.memory.prototypes_per_class} protos/class)",
            f"  Attention:     {self.attention.num_heads} heads (key_dim={self.attention.key_dim})",
            f"  Fusion dim:    {self.fusion.fusion_dim}",
            f"  Mixed prec:    {self.training.mixed_precision}",
            f"  Seed:          {self.training.seed}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ─── Default config instance ────────────────────────────────────────────────

def get_default_config() -> MAVISConfig:
    """Return the default MAVIS configuration with spec §47 initial values."""
    return MAVISConfig()


if __name__ == "__main__":
    config = get_default_config()
    print(config.summary())
    # Save default configs
    config.save(os.path.join(CONFIGS_DIR, "mavis.yaml"))
    
    # Save baseline config (no EWC, no memory)
    baseline = get_default_config()
    baseline.ewc.enabled = False
    baseline.save(os.path.join(CONFIGS_DIR, "baseline.yaml"))
    
    # Save EWC-only config
    ewc_config = get_default_config()
    ewc_config.save(os.path.join(CONFIGS_DIR, "ewc.yaml"))
    
    # Save memory-only config (no EWC)
    mem_config = get_default_config()
    mem_config.ewc.enabled = False
    mem_config.save(os.path.join(CONFIGS_DIR, "memory.yaml"))
