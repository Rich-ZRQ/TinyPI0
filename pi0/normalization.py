"""State and action normalization used by pi0."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class NormStats:
    """Per-dimension dataset statistics."""

    mean: Tensor
    std: Tensor
    q01: Tensor | None = None
    q99: Tensor | None = None

    def __post_init__(self) -> None:
        tensors = {
            "mean": self.mean,
            "std": self.std,
        }

        if self.q01 is not None:
            tensors["q01"] = self.q01

        if self.q99 is not None:
            tensors["q99"] = self.q99

        for name, tensor in tensors.items():
            if tensor.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional, got {tuple(tensor.shape)}")

            if not tensor.is_floating_point():
                raise TypeError(f"{name} must be floating point")

            if tensor.shape != self.mean.shape:
                raise ValueError(f"{name} must have shape {tuple(self.mean.shape)}, got {tuple(tensor.shape)}")

        if torch.any(self.std < 0):
            raise ValueError("std cannot contain negative values")

        if (self.q01 is None) != (self.q99 is None):
            raise ValueError("q01 and q99 must be provided together")


def compute_norm_stats(values: Tensor) -> NormStats:
    """Compute exact statistics over every dimension except the last."""

    if values.ndim < 2:
        raise ValueError("values must have at least two dimensions")

    if not values.is_floating_point():
        raise TypeError(f"values must be floating point, got {values.dtype}")

    flattened = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1, values.shape[-1])

    if flattened.shape[0] < 2:
        raise ValueError("at least two vectors are required")

    quantiles = torch.quantile(
        flattened,
        torch.tensor(
            [0.01, 0.99],
            dtype=torch.float64,
        ),
        dim=0,
    )

    return NormStats(
        mean=flattened.mean(dim=0).to(torch.float32),
        std=flattened.std(
            dim=0,
            correction=0,
        ).to(torch.float32),
        q01=quantiles[0].to(torch.float32),
        q99=quantiles[1].to(torch.float32),
    )


class Pi0Normalizer(nn.Module):
    """Normalize state/actions and invert predicted actions."""

    def __init__(
        self,
        state_stats: NormStats,
        action_stats: NormStats,
        *,
        use_quantiles: bool = False,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()

        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        if use_quantiles and (
            state_stats.q01 is None or state_stats.q99 is None or action_stats.q01 is None or action_stats.q99 is None
        ):
            raise ValueError("quantile normalization requires q01 and q99")

        self.use_quantiles = use_quantiles
        self.epsilon = epsilon

        self._register_stats("state", state_stats)
        self._register_stats("action", action_stats)

    def _register_stats(
        self,
        prefix: str,
        stats: NormStats,
    ) -> None:
        self.register_buffer(
            f"{prefix}_mean",
            stats.mean.detach().clone().to(torch.float32),
        )
        self.register_buffer(
            f"{prefix}_std",
            stats.std.detach().clone().to(torch.float32),
        )
        self.register_buffer(
            f"{prefix}_q01",
            (None if stats.q01 is None else stats.q01.detach().clone().to(torch.float32)),
        )
        self.register_buffer(
            f"{prefix}_q99",
            (None if stats.q99 is None else stats.q99.detach().clone().to(torch.float32)),
        )

    def normalize_state(self, state: Tensor) -> Tensor:
        return self._transform(
            state,
            mean=self.state_mean,
            std=self.state_std,
            q01=self.state_q01,
            q99=self.state_q99,
            inverse=False,
        )

    def normalize_actions(self, actions: Tensor) -> Tensor:
        return self._transform(
            actions,
            mean=self.action_mean,
            std=self.action_std,
            q01=self.action_q01,
            q99=self.action_q99,
            inverse=False,
        )

    def unnormalize_actions(self, actions: Tensor) -> Tensor:
        return self._transform(
            actions,
            mean=self.action_mean,
            std=self.action_std,
            q01=self.action_q01,
            q99=self.action_q99,
            inverse=True,
        )

    def _transform(
        self,
        values: Tensor,
        *,
        mean: Tensor,
        std: Tensor,
        q01: Tensor | None,
        q99: Tensor | None,
        inverse: bool,
    ) -> Tensor:
        if not values.is_floating_point():
            raise TypeError(f"values must be floating point, got {values.dtype}")

        stats_dim = min(
            values.shape[-1],
            mean.shape[0],
        )

        active_values = values[..., :stats_dim]

        if self.use_quantiles:
            assert q01 is not None
            assert q99 is not None

            lower = q01[:stats_dim].to(
                device=values.device,
                dtype=values.dtype,
            )
            upper = q99[:stats_dim].to(
                device=values.device,
                dtype=values.dtype,
            )

            scale = upper - lower + self.epsilon

            if inverse:
                transformed = (active_values + 1.0) / 2.0 * scale + lower
            else:
                transformed = (active_values - lower) / scale * 2.0 - 1.0
        else:
            active_mean = mean[:stats_dim].to(
                device=values.device,
                dtype=values.dtype,
            )
            active_std = std[:stats_dim].to(
                device=values.device,
                dtype=values.dtype,
            )

            if inverse:
                transformed = active_values * (active_std + self.epsilon) + active_mean
            else:
                transformed = (active_values - active_mean) / (active_std + self.epsilon)

        if stats_dim == values.shape[-1]:
            return transformed

        # 超出真实机器人维度的padding维保持不变。
        return torch.cat(
            [
                transformed,
                values[..., stats_dim:],
            ],
            dim=-1,
        )


def save_norm_stats(
    path: str | Path,
    stats: Mapping[str, NormStats],
) -> None:
    """Save statistics as portable JSON."""

    output: dict[str, dict[str, list[float] | None]] = {}

    for key, value in stats.items():
        output[key] = {
            "mean": value.mean.tolist(),
            "std": value.std.tolist(),
            "q01": (None if value.q01 is None else value.q01.tolist()),
            "q99": (None if value.q99 is None else value.q99.tolist()),
        }

    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    destination.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )


def load_norm_stats(
    path: str | Path,
) -> dict[str, NormStats]:
    """Load statistics from JSON."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    result: dict[str, NormStats] = {}

    for key, value in raw.items():
        result[key] = NormStats(
            mean=torch.tensor(
                value["mean"],
                dtype=torch.float32,
            ),
            std=torch.tensor(
                value["std"],
                dtype=torch.float32,
            ),
            q01=(
                None
                if value["q01"] is None
                else torch.tensor(
                    value["q01"],
                    dtype=torch.float32,
                )
            ),
            q99=(
                None
                if value["q99"] is None
                else torch.tensor(
                    value["q99"],
                    dtype=torch.float32,
                )
            ),
        )

    return result
