"""Training settings shared by local smoke tests and server runs."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Optimizer, validation and checkpoint settings for one training run."""

    output_dir: Path
    max_steps: int = 30_000
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2.5e-5
    end_learning_rate: float = 2.5e-6
    warmup_steps: int = 1_000
    decay_steps: int = 30_000
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 1e-10
    max_grad_norm: float = 1.0
    validation_fraction: float = 0.1
    validation_interval: int = 500
    validation_batches: int = 32
    checkpoint_interval: int = 1_000
    num_workers: int = 4
    seed: int = 42
    gradient_checkpointing: bool = False
    compile_model: bool = False

    def __post_init__(self) -> None:
        positive_integer_fields = (
            "max_steps",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "warmup_steps",
            "decay_steps",
            "validation_interval",
            "validation_batches",
            "checkpoint_interval",
        )

        for name in positive_integer_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")

        for name in ("learning_rate", "end_learning_rate", "epsilon", "max_grad_norm"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

        if self.weight_decay < 0:
            raise ValueError(f"weight_decay cannot be negative, got {self.weight_decay}")

        if not 0 < self.validation_fraction < 1:
            raise ValueError(f"validation_fraction must be between zero and one, got {self.validation_fraction}")

        for name in ("beta1", "beta2"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1), got {getattr(self, name)}")

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps


SO101_4090_TRAINING = TrainingConfig(
    output_dir=Path("checkpoints/so101_recommended"),
)
