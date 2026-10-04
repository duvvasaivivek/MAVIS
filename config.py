"""
MAVIS Configuration Module
===========================
Central configuration for all hyperparameters and paths.
Initial values from spec §47 — these are starting values, NOT final optimized values.
"""

import os
import yaml
from dataclasses import dataclass, field, asdict


# ─── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
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
    # Optional (enable after baseline is stable)
    random_erasing: bool = False
    mixup: bool = False
    cutmix: bool = False


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 256
    epochs_per_task: int = 100
    learning_rate: float = 0.005
    optimizer: str = "adam"
    mixed_precision: bool = True
    seed: int = 42
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
    hidden_dim: int = 256
    num_classes: int = 200
    dropout_rate: float = 0.3


@dataclass
class EWCConfig:
    """
    Elastic Weight Consolidation — LOSS-LEVEL regularization.
    
    EWC adds a penalty to the loss function during backpropagation:
        L_total = L_CE + (lambda/2) * Σ F_i * (θ_i − θ_i*)²
    
    It does NOT appear in the forward pass / data flow.
    Fisher values and old parameters are frozen reference tensors.
    """
    enabled: bool = True
    ewc_lambda: float = 50000.0
    fisher_samples: int = 1000
    # Lambda sweep values for experiments
    lambda_sweep: list = field(default_factory=lambda: [10, 100, 1000, 5000])


@dataclass
class MemoryConfig:
    """
    External Neural Memory — stores compact feature representations.
    
    Each entry: (512-D feature, class label, importance score, task_id)
    Memory is fixed-size; adaptive selection replaces least useful entries.
    """
    capacity: int = 500
    feature_dim: int = 512
    strategy: str = "adaptive"  # "adaptive", "random", "fifo"
    # Importance scoring weights
    novelty_weight: float = 1.0
    diversity_weight: float = 0.5
    class_balance_weight: float = 0.5
    # Budget sweep for experiments
    capacity_sweep: list = field(default_factory=lambda: [50, 100, 250, 500, 1000])


@dataclass
class AttentionConfig:
    """
    Multi-Head Attention for memory retrieval.
    
    Operates in the forward pass:
        Q = W_Q · f_current
        K = W_K · M_memory
        V = W_V · M_memory
        Retrieved = softmax(QK^T / √d_k) · V
    """
    num_heads: int = 4
    key_dim: int = 256
    query_dim: int = 256
    value_dim: int = 512
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
                setattr(config, section_name, section_cls(**section_data))
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
            f"  EWC:           {'ON (λ=' + str(self.ewc.ewc_lambda) + ')' if self.ewc.enabled else 'OFF'}",
            f"  Memory:        {self.memory.capacity} entries ({self.memory.strategy})",
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
