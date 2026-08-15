"""Rotary position embedding used by Gemma attention."""

import torch
from torch import Tensor, nn


def rotate_half(x: Tensor) -> Tensor:
    """Rotate pairs represented by the two halves of a vector."""

    if x.shape[-1] % 2 != 0:
        raise ValueError(f"the last dimension must be even, got {x.shape[-1]}")

    half = x.shape[-1] // 2

    first_half = x[..., :half]
    second_half = x[..., half:]

    return torch.cat(
        [
            -second_half,
            first_half,
        ],
        dim=-1,
    )  # 用来对应sin前面的向量参数，即：result = (x1, x2)cos + (-x2, x1)sin 其中 (x2, x1)就是rotate_half的结果


class GemmaRotaryEmbedding(nn.Module):
    """Generate Gemma-compatible rotary sine/cosine values."""

    def __init__(
        self,
        head_dim: int,
        *,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()

        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")

        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")

        if base <= 0:
            raise ValueError(f"base must be positive, got {base}")

        self.head_dim = head_dim
        self.base = base

        dimension_indices = torch.arange(
            0,
            head_dim,
            2,
            dtype=torch.float32,
        )

        inverse_frequency = 1.0 / (base ** (dimension_indices / head_dim))

        self.register_buffer(
            "inverse_frequency",
            inverse_frequency,
            persistent=False,
        )

    @torch.no_grad()
    def forward(
        self,
        x: Tensor,
        position_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not x.is_floating_point():
            raise TypeError(f"x must be floating point, got {x.dtype}")

        if position_ids.ndim != 2:
            raise ValueError(f"position_ids must have shape [B, S], got {tuple(position_ids.shape)}")

        if position_ids.is_floating_point():
            raise TypeError(f"position_ids must use an integer dtype, got {position_ids.dtype}")

        if x.shape[0] != position_ids.shape[0]:
            raise ValueError("x and position_ids must have the same batch size")

        if x.shape[-2] != position_ids.shape[1]:
            raise ValueError("x sequence length and position_ids length must match")

        if x.device != position_ids.device:
            raise ValueError("x and position_ids must be on the same device")

        """
                频率 θ (D/2个) →
                θ₀    θ₁    θ₂  ...
        位置0 [ 0·θ₀  0·θ₁  0·θ₂ ]   ← token0,位置0,全是0(不转)
        位置1 [ 1·θ₀  1·θ₁  1·θ₂ ]   ← token1,每个平面转 1·θᵢ
        位置2 [ 2·θ₀  2·θ₁  2·θ₂ ]   ← token2,转 2·θᵢ
        ↑
        seq维

        每个元素 = 该token在该平面的旋转角度
        """
        frequencies = torch.einsum(
            "bs,d->bsd",
            position_ids.to(torch.float32),
            self.inverse_frequency.to(
                device=x.device,
                dtype=torch.float32,
            ),
        )

        """
        doubled_frequencies = [ θ₀ θ₁ ... θ_{D/2-1} │ θ₀ θ₁ ... θ_{D/2-1} ]
                          └──── 前半 ────┘  └──── 后半(复制)──┘
                            第i维              第i+D/2维
                            └──── 用同一个角度 ────┘
        """
        doubled_frequencies = torch.cat(
            [
                frequencies,
                frequencies,
            ],
            dim=-1,
        )

        cosine = torch.cos(doubled_frequencies)
        sine = torch.sin(doubled_frequencies)

        return (
            cosine.to(dtype=x.dtype),
            sine.to(dtype=x.dtype),
        )


def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cosine: Tensor,
    sine: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply RoPE to query and key tensors.

    Query and key use [B, H, S, D].
    Cosine and sine use [B, S, D].
    """

    if query.ndim != 4:
        raise ValueError(f"query must have shape [B, H, S, D], got {tuple(query.shape)}")

    if key.ndim != 4:
        raise ValueError(f"key must have shape [B, H, S, D], got {tuple(key.shape)}")

    expected_rotary_shape = (
        query.shape[0],
        query.shape[2],
        query.shape[3],
    )

    if cosine.shape != expected_rotary_shape:
        raise ValueError(f"cosine must have shape {expected_rotary_shape}, got {tuple(cosine.shape)}")

    if sine.shape != expected_rotary_shape:
        raise ValueError(f"sine must have shape {expected_rotary_shape}, got {tuple(sine.shape)}")

    if key.shape[0] != query.shape[0]:
        raise ValueError("query and key batch sizes must match")

    if key.shape[2:] != query.shape[2:]:
        raise ValueError("query and key sequence/head dimensions must match")

    cosine = cosine[:, None, :, :]
    sine = sine[:, None, :, :]

    rotated_query = query * cosine + rotate_half(query) * sine

    rotated_key = key * cosine + rotate_half(key) * sine

    return rotated_query, rotated_key
