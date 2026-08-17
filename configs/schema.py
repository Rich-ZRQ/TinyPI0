"""Architecture configuration definitions for Tiny pi0."""

from dataclasses import dataclass
from typing import Literal

DTypeName = Literal["float32", "bfloat16"]


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration of a Gemma transformer expert."""

    width: int  # 每个 token 的隐藏状态维度
    depth: int  # Transformer 层数。
    mlp_dim: int  # 每层 MLP 中间维度
    num_heads: int  # Query Attention head 数。
    num_kv_heads: int  # 标准自注意力头num_heads == num_kv_heads，但是Gemma允许1 <= num_kv_heads < num_heads,query 头被分成若干组,每组共享一组 K/V。 官方使用 GQA，值为 1
    head_dim: int

    def __post_init__(self) -> None:
        positive_fields = (
            "width",
            "depth",
            "mlp_dim",
            "num_heads",
            "num_kv_heads",
            "head_dim",
        )

        for field_name in positive_fields:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_heads must be divisible by num_kv_heads, "
                f"got num_heads={self.num_heads}, "
                f"num_kv_heads={self.num_kv_heads}"
            )


@dataclass(frozen=True)
class VisionConfig:
    """Configuration of the SigLIP vision encoder."""

    image_size: int
    patch_size: int
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    projection_dim: int
    num_channels: int = 3

    def __post_init__(self) -> None:
        positive_fields = (
            "image_size",
            "patch_size",
            "width",  # width 是 SigLIP 内部 token 维度
            "depth",
            "mlp_dim",
            "num_heads",
            "projection_dim",
            "num_channels",
        )

        for field_name in positive_fields:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")

        if self.image_size % self.patch_size != 0:
            raise ValueError(
                "image_size must be divisible by patch_size, "
                f"got image_size={self.image_size}, "
                f"patch_size={self.patch_size}"
            )

        if self.width % self.num_heads != 0:
            raise ValueError(
                f"vision width must be divisible by num_heads, got width={self.width}, num_heads={self.num_heads}"
            )

    @property
    def patches_per_side(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_tokens(self) -> int:
        return self.patches_per_side**2


@dataclass(frozen=True)
class Pi0Config:
    """Complete architecture configuration for pi0."""

    vision: VisionConfig
    paligemma: TransformerConfig
    action_expert: TransformerConfig

    vocab_size: int = 257_152  # 257152
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48
    dtype: DTypeName = "float32"

    def __post_init__(self) -> None:
        positive_fields = (
            "vocab_size",
            "action_dim",
            "action_horizon",
            "max_token_len",
        )

        for field_name in positive_fields:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")

        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError(f"dtype must be 'float32' or 'bfloat16', got {self.dtype!r}")

        if self.vision.projection_dim != self.paligemma.width:
            raise ValueError(
                "vision projection_dim must equal paligemma width, "
                f"got projection_dim={self.vision.projection_dim}, "
                f"paligemma_width={self.paligemma.width}"
            )

        shared_attention_fields = (
            "depth",
            "num_heads",
            "num_kv_heads",
            "head_dim",
        )

        for field_name in shared_attention_fields:
            paligemma_value = getattr(self.paligemma, field_name)
            action_expert_value = getattr(
                self.action_expert,
                field_name,
            )

            if paligemma_value != action_expert_value:
                raise ValueError(
                    f"paligemma.{field_name} must equal "
                    f"action_expert.{field_name}, "
                    f"got {paligemma_value} and "
                    f"{action_expert_value}"
                )
