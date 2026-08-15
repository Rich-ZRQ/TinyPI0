"""Trainable pi0 core operating on pre-embedded prefix tokens."""

import torch
from torch import Tensor, nn

from configs.schema import Pi0Config
from pi0.action_embedding import Pi0ActionEmbedding
from pi0.attention_mask import make_att_2d_masks
from pi0.flow_matching import (
    flow_matching_loss,
    make_flow_matching_target,
    sample_noise,
    sample_time,
)
from pi0.joint_transformer import JointTransformer, PrefixKVCache
from pi0.types import validate_actions


class Pi0Core(nn.Module):
    """Connect prefix tokens, action tokens and flow matching."""

    def __init__(self, config: Pi0Config) -> None:
        super().__init__()

        self.config = config

        self.action_embedding = Pi0ActionEmbedding(config)
        self.transformer = JointTransformer(config)

    def predict_velocity(
        self,
        prefix_tokens: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        state: Tensor,
        noisy_actions: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        """Predict flow velocity from an embedded prefix."""

        self._validate_prefix(
            prefix_tokens,
            prefix_pad_masks,
            prefix_att_masks,
            state,
        )

        (
            suffix_tokens,
            suffix_pad_masks,
            suffix_att_masks,
        ) = self.action_embedding(
            state,
            noisy_actions,
            timestep,
        )

        pad_masks = torch.cat(
            [
                prefix_pad_masks,
                suffix_pad_masks,
            ],
            dim=1,
        )

        att_masks = torch.cat(
            [
                prefix_att_masks,
                suffix_att_masks,
            ],
            dim=1,
        )

        attention_mask = make_att_2d_masks(
            pad_masks,
            att_masks,
        )

        position_ids = (
            torch.cumsum(
                pad_masks.to(torch.int64),
                dim=1,
            )
            - 1
        )

        _, suffix_output = self.transformer(
            prefix_tokens,
            suffix_tokens,
            position_ids,
            attention_mask,
        )

        action_output = suffix_output[
            :,
            -self.config.action_horizon :,
        ]

        return self.action_embedding.project_velocity(action_output)

    def training_loss(
        self,
        prefix_tokens: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        state: Tensor,
        actions: Tensor,
        *,
        noise: Tensor | None = None,
        timestep: Tensor | None = None,
    ) -> Tensor:
        """Return unreduced official pi0 flow-matching loss."""

        validate_actions(
            actions,
            config=self.config,
            batch_size=state.shape[0],
        )

        if actions.device != state.device:
            raise ValueError("actions and state must be on the same device")

        if noise is None:
            noise = sample_noise(
                tuple(actions.shape),
                device=actions.device,
                dtype=actions.dtype,
            )

        if timestep is None:
            timestep = sample_time(
                actions.shape[0],
                device=actions.device,
            )

        noisy_actions, target_velocity = make_flow_matching_target(
            actions,
            noise,
            timestep,
        )

        predicted_velocity = self.predict_velocity(
            prefix_tokens,
            prefix_pad_masks,
            prefix_att_masks,
            state,
            noisy_actions,
            timestep,
        )

        return flow_matching_loss(
            predicted_velocity,
            target_velocity,
        )

    def prefill_prefix(
        self,
        prefix_tokens: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        state: Tensor,
    ) -> PrefixKVCache:
        """Build the per-layer prefix K/V cache used during sampling."""

        self._validate_prefix(
            prefix_tokens,
            prefix_pad_masks,
            prefix_att_masks,
            state,
        )

        prefix_attention_mask = make_att_2d_masks(
            prefix_pad_masks,
            prefix_att_masks,
        )

        prefix_position_ids = (
            torch.cumsum(
                prefix_pad_masks.to(torch.int64),
                dim=1,
            )
            - 1
        )

        _, prefix_cache = self.transformer.prefill_prefix(
            prefix_hidden_states=prefix_tokens,
            position_ids=prefix_position_ids,
            attention_mask=prefix_attention_mask,
        )

        return prefix_cache

    def predict_velocity_with_cache(
        self,
        prefix_cache: PrefixKVCache,
        prefix_pad_masks: Tensor,
        state: Tensor,
        noisy_actions: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        """Predict velocity while reusing precomputed prefix K/V."""

        (
            suffix_tokens,
            suffix_pad_masks,
            suffix_att_masks,
        ) = self.action_embedding(
            state,
            noisy_actions,
            timestep,
        )

        suffix_attention_mask = make_att_2d_masks(
            suffix_pad_masks,
            suffix_att_masks,
        )

        prefix_attention_mask = prefix_pad_masks[:, None, :].expand(
            -1,
            suffix_tokens.shape[1],
            -1,
        )

        attention_mask = torch.cat(
            [
                prefix_attention_mask,
                suffix_attention_mask,
            ],
            dim=-1,
        )

        prefix_valid_length = prefix_pad_masks.to(torch.int64).sum(
            dim=1,
            keepdim=True,
        )
        suffix_position_ids = (
            prefix_valid_length
            + torch.cumsum(
                suffix_pad_masks.to(torch.int64),
                dim=1,
            )
            - 1
        )

        suffix_output = self.transformer.forward_suffix_with_cache(
            suffix_hidden_states=suffix_tokens,
            prefix_cache=prefix_cache,
            position_ids=suffix_position_ids,
            attention_mask=attention_mask,
        )

        action_output = suffix_output[
            :,
            -self.config.action_horizon :,
        ]

        return self.action_embedding.project_velocity(action_output)

    def _validate_prefix(
        self,
        prefix_tokens: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        state: Tensor,
    ) -> None:
        if prefix_tokens.ndim != 3:
            raise ValueError(
                f"prefix_tokens must have shape [B, prefix_length, paligemma_width], got {tuple(prefix_tokens.shape)}"
            )

        batch_size, prefix_length, width = prefix_tokens.shape

        if width != self.config.paligemma.width:
            raise ValueError(f"prefix token width must equal {self.config.paligemma.width}, got {width}")

        expected_mask_shape = (
            batch_size,
            prefix_length,
        )

        if prefix_pad_masks.shape != expected_mask_shape:
            raise ValueError(
                f"prefix_pad_masks must have shape {expected_mask_shape}, got {tuple(prefix_pad_masks.shape)}"
            )

        if prefix_att_masks.shape != expected_mask_shape:
            raise ValueError(
                f"prefix_att_masks must have shape {expected_mask_shape}, got {tuple(prefix_att_masks.shape)}"
            )

        if state.shape[0] != batch_size:
            raise ValueError("prefix_tokens and state must have the same batch size")

        if not prefix_tokens.is_floating_point():
            raise TypeError(f"prefix_tokens must be floating point, got {prefix_tokens.dtype}")

        if prefix_pad_masks.dtype != torch.bool:
            raise TypeError(f"prefix_pad_masks must be bool, got {prefix_pad_masks.dtype}")

        if prefix_att_masks.dtype != torch.bool:
            raise TypeError(f"prefix_att_masks must be bool, got {prefix_att_masks.dtype}")

        tensors = {
            "prefix_pad_masks": prefix_pad_masks,
            "prefix_att_masks": prefix_att_masks,
            "state": state,
        }

        for name, tensor in tensors.items():
            if tensor.device != prefix_tokens.device:
                raise ValueError(f"{name} and prefix_tokens must be on the same device")
