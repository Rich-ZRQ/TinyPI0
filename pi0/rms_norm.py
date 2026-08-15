"""Gemma-compatible RMS normalization."""

import torch
from torch import Tensor, nn


class GemmaRMSNorm(nn.Module):
    """RMSNorm using the parameter convention from openpi Gemma."""

    def __init__(self, dim: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.dim = dim
        self.eps = eps

        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {x.shape[-1]}")

        if not x.is_floating_point():
            raise TypeError(f"x must be floating point, got {x.dtype}")

        input_dtype = x.dtype

        x_float = x.to(torch.float32)

        mean_square = torch.mean(
            torch.square(x_float),
            dim=-1,
            keepdim=True,
        )

        normalized = x_float * torch.rsqrt(mean_square + self.eps)

        output = normalized * (1.0 + self.weight.to(torch.float32))

        return output.to(dtype=input_dtype)
