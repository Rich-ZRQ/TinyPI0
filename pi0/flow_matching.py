"""Flow-matching utilities used to train pi0."""

import torch
from torch import Tensor
from torch.nn import functional as F


def sample_noise(
    shape: tuple[int, ...],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample standard Gaussian action noise."""

    if len(shape) != 3:
        raise ValueError(f"noise shape must be [B, action_horizon, action_dim], got {shape}")

    if any(dimension <= 0 for dimension in shape):
        raise ValueError(f"all noise dimensions must be positive, got {shape}")

    return torch.randn(
        shape,
        dtype=dtype,
        device=device,
        generator=generator,
    )  # noise ~ N(0, 1)


def sample_time(
    batch_size: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample official pi0 flow timesteps."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    # Beta(alpha=1.5, beta=1) has inverse CDF u ** (1 / alpha).
    # This exact form lets the trainer use a private, checkpointable Generator.
    uniform = torch.rand(
        batch_size,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    beta_time = uniform ** (1.0 / 1.5)

    time = beta_time * 0.999 + 0.001  # 把范围从[0, 1]映射到[0.001, 0.999]，避免采样到极端值

    return time.to(
        dtype=torch.float32,
        device=device,
    )


def make_flow_matching_target(
    actions: Tensor,
    noise: Tensor,
    timestep: Tensor,
) -> tuple[Tensor, Tensor]:
    """Construct x_t and its target flow velocity.

    The pi0 path is:

        x_t = t * noise + (1 - t) * actions
        u_t = noise - actions
    """

    _validate_flow_inputs(
        actions,
        noise,
        timestep,
    )

    expanded_time = timestep.to(dtype=actions.dtype)[:, None, None]

    noisy_actions = expanded_time * noise + (1.0 - expanded_time) * actions

    target_velocity = noise - actions

    return noisy_actions, target_velocity


def flow_matching_loss(
    predicted_velocity: Tensor,
    target_velocity: Tensor,
) -> Tensor:
    """Return the unreduced pi0 velocity MSE loss."""

    if predicted_velocity.shape != target_velocity.shape:
        raise ValueError(
            "predicted_velocity and target_velocity must have "
            "the same shape, "
            f"got {tuple(predicted_velocity.shape)} and "
            f"{tuple(target_velocity.shape)}"
        )

    if predicted_velocity.ndim != 3:
        raise ValueError(
            f"velocity tensors must have shape [B, action_horizon, action_dim], got {tuple(predicted_velocity.shape)}"
        )

    if not predicted_velocity.is_floating_point():
        raise TypeError(f"predicted_velocity must be floating point, got {predicted_velocity.dtype}")

    if not target_velocity.is_floating_point():
        raise TypeError(f"target_velocity must be floating point, got {target_velocity.dtype}")

    if predicted_velocity.device != target_velocity.device:
        raise ValueError("predicted_velocity and target_velocity must be on the same device")

    return F.mse_loss(
        predicted_velocity,
        target_velocity,
        reduction="none",
    )


def _validate_flow_inputs(
    actions: Tensor,
    noise: Tensor,
    timestep: Tensor,
) -> None:
    if actions.ndim != 3:
        raise ValueError(f"actions must have shape [B, action_horizon, action_dim], got {tuple(actions.shape)}")

    if noise.shape != actions.shape:
        raise ValueError(
            f"noise must have the same shape as actions, got {tuple(noise.shape)} and {tuple(actions.shape)}"
        )

    expected_time_shape = (actions.shape[0],)

    if timestep.shape != expected_time_shape:
        raise ValueError(f"timestep must have shape {expected_time_shape}, got {tuple(timestep.shape)}")

    tensors = {
        "actions": actions,
        "noise": noise,
        "timestep": timestep,
    }

    for name, tensor in tensors.items():
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be floating point, got {tensor.dtype}")

        if tensor.device != actions.device:
            raise ValueError(f"{name} and actions must be on the same device")
