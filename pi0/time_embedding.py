"""Sinusoidal timestep embedding used by pi0 flow matching."""

import math

import torch
from torch import Tensor


def create_sinusoidal_pos_embedding(
    time: Tensor,
    dimension: int,
    *,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> Tensor:
    """Encode scalar flow timesteps as sinusoidal vectors.

    Args:
        time:
            Floating-point tensor with shape [B]. Values normally
            lie in [0, 1].

        dimension:
            Output embedding dimension. It must be even because
            half the dimensions are sine values and half are cosine
            values.

        min_period:
            Smallest sinusoidal period.

        max_period:
            Largest sinusoidal period.

    Returns:
        Tensor with shape [B, dimension].
    """

    if time.ndim != 1:
        raise ValueError(f"time must have shape [B], got {tuple(time.shape)}")

    if not time.is_floating_point():
        raise TypeError(f"time must be floating point, got {time.dtype}")

    if dimension <= 0:
        raise ValueError(f"dimension must be positive, got {dimension}")

    if dimension % 2 != 0:
        raise ValueError(f"dimension must be even, got {dimension}")

    if min_period <= 0:
        raise ValueError(f"min_period must be positive, got {min_period}")

    if max_period < min_period:
        raise ValueError(f"max_period must be greater than or equal to min_period, got {max_period} and {min_period}")

    half_dimension = dimension // 2

    compute_dtype = torch.float64

    fraction = torch.linspace(
        0.0,
        1.0,
        half_dimension,
        dtype=compute_dtype,
        device=time.device,
    )

    period = min_period * (max_period / min_period) ** fraction

    angular_frequency = 2.0 * math.pi / period

    sinusoid_input = time.to(compute_dtype)[:, None] * angular_frequency[None, :]

    embedding = torch.cat(
        [
            torch.sin(sinusoid_input),
            torch.cos(sinusoid_input),
        ],
        dim=-1,
    )

    return embedding.to(dtype=time.dtype)
