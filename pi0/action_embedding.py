"""State, action and timestep embedding used by pi0."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from configs.schema import Pi0Config
from pi0.time_embedding import (
    create_sinusoidal_pos_embedding,
)


class Pi0ActionEmbedding(nn.Module):
    """Convert pi0 state and flow inputs into action-expert tokens."""

    def __init__(self, config: Pi0Config) -> None:
        super().__init__()

        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.width = config.action_expert.width

        self.state_proj = nn.Linear(
            self.action_dim,
            self.width,
        )

        self.action_in_proj = nn.Linear(
            self.action_dim,
            self.width,
        )

        self.action_time_mlp_in = nn.Linear(
            2 * self.width,
            self.width,
        )

        self.action_time_mlp_out = nn.Linear(
            self.width,
            self.width,
        )

        self.action_out_proj = nn.Linear(
            self.width,
            self.action_dim,
        )

    def forward(
        self,
        state: Tensor,
        noisy_actions: Tensor,
        timestep: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build state and action suffix tokens and their masks."""

        self._validate_inputs(
            state,
            noisy_actions,
            timestep,
        )

        state_token = self.state_proj(state)[:, None, :]

        action_embedding = self.action_in_proj(noisy_actions)

        time_embedding = create_sinusoidal_pos_embedding(
            timestep,
            self.width,
            min_period=4e-3,
            max_period=4.0,
        )

        time_embedding = time_embedding.to(dtype=action_embedding.dtype)

        time_embedding = time_embedding[:, None, :].expand(
            -1,
            self.action_horizon,
            -1,
        )

        action_time_embedding = torch.cat(
            [
                action_embedding,
                time_embedding,
            ],
            dim=-1,
        )

        action_tokens = self.action_time_mlp_in(action_time_embedding)
        action_tokens = F.silu(action_tokens)
        action_tokens = self.action_time_mlp_out(action_tokens)

        suffix_tokens = torch.cat(
            [
                state_token,
                action_tokens,
            ],
            dim=1,
        )

        batch_size = state.shape[0]
        suffix_length = 1 + self.action_horizon

        pad_masks = torch.ones(
            batch_size,
            suffix_length,
            dtype=torch.bool,
            device=state.device,
        )

        att_masks = torch.zeros(
            batch_size,
            suffix_length,
            dtype=torch.bool,
            device=state.device,
        )

        att_masks[:, 0] = True
        att_masks[:, 1] = True

        return (
            suffix_tokens,
            pad_masks,
            att_masks,
        )

    def project_velocity(
        self,
        action_hidden_states: Tensor,
    ) -> Tensor:
        """Project action-expert outputs to flow velocity."""

        expected_suffix = (
            self.action_horizon,
            self.width,
        )

        if action_hidden_states.ndim != 3:
            raise ValueError(
                "action_hidden_states must have shape "
                "[B, action_horizon, width], "
                f"got {tuple(action_hidden_states.shape)}"
            )

        if action_hidden_states.shape[1:] != expected_suffix:
            raise ValueError(
                "action_hidden_states must have trailing shape "
                f"{expected_suffix}, "
                f"got {tuple(action_hidden_states.shape[1:])}"
            )

        if not action_hidden_states.is_floating_point():
            raise TypeError(f"action_hidden_states must be floating point, got {action_hidden_states.dtype}")

        return self.action_out_proj(action_hidden_states)

    def _validate_inputs(
        self,
        state: Tensor,
        noisy_actions: Tensor,
        timestep: Tensor,
    ) -> None:
        if state.ndim != 2:
            raise ValueError(f"state must have shape [B, action_dim], got {tuple(state.shape)}")

        batch_size = state.shape[0]

        expected_state_shape = (
            batch_size,
            self.action_dim,
        )
        expected_action_shape = (
            batch_size,
            self.action_horizon,
            self.action_dim,
        )
        expected_time_shape = (batch_size,)

        if state.shape != expected_state_shape:
            raise ValueError(f"state must have shape {expected_state_shape}, got {tuple(state.shape)}")

        if noisy_actions.shape != expected_action_shape:
            raise ValueError(f"noisy_actions must have shape {expected_action_shape}, got {tuple(noisy_actions.shape)}")

        if timestep.shape != expected_time_shape:
            raise ValueError(f"timestep must have shape {expected_time_shape}, got {tuple(timestep.shape)}")

        tensors = {
            "state": state,
            "noisy_actions": noisy_actions,
            "timestep": timestep,
        }

        for name, tensor in tensors.items():
            if not tensor.is_floating_point():
                raise TypeError(f"{name} must be floating point, got {tensor.dtype}")

            if tensor.device != state.device:
                raise ValueError(f"{name} and state must be on the same device")
