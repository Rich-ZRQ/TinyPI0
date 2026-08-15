"""End-to-end Tiny pi0 policy."""

from dataclasses import replace

import torch
from torch import Tensor, nn

from configs.schema import Pi0Config
from pi0.core import Pi0Core
from pi0.normalization import Pi0Normalizer
from pi0.paligemma_prefix import PaliGemmaPrefixEncoder
from pi0.prefix_embedding import Pi0PrefixEmbedding
from pi0.types import Actions, Observation


class Pi0Policy(nn.Module):
    """Connect frozen prefix encoding, dual experts and flow sampling."""

    def __init__(
        self,
        config: Pi0Config,
        prefix_encoder: PaliGemmaPrefixEncoder,
        normalizer: Pi0Normalizer | None = None,
    ) -> None:
        super().__init__()

        self.config = config

        self.prefix_embedding = Pi0PrefixEmbedding(
            config=config,
            prefix_encoder=prefix_encoder,
        )

        reference_parameter = next(prefix_encoder.parameters())
        trainable_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[config.dtype]

        self.core = Pi0Core(config).to(
            device=reference_parameter.device,
            dtype=trainable_dtype,
        )
        self._initialize_trainable_weights()

        self.normalizer = normalizer

        if self.normalizer is not None:
            self.normalizer.to(
                device=reference_parameter.device,
            )

    def _initialize_trainable_weights(self, initializer_std: float = 0.02) -> None:
        """Apply Gemma's official initialization to the from-scratch modules."""

        modules = (
            self.prefix_embedding.input_projection,
            *self.core.modules(),
        )

        with torch.no_grad():
            for module in modules:
                if not isinstance(module, nn.Linear):
                    continue

                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=initializer_std,
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            # This expert has no shape-compatible pretrained checkpoint. A zero
            # velocity head keeps the first flow updates stable while preserving
            # the exact architecture and lets gradients enter the backbone as
            # soon as the head moves away from zero.
            nn.init.zeros_(self.core.action_embedding.action_out_proj.weight)
            nn.init.zeros_(self.core.action_embedding.action_out_proj.bias)

    @property
    def model_device(self) -> torch.device:
        return next(self.core.parameters()).device

    @property
    def model_dtype(self) -> torch.dtype:
        return next(self.core.parameters()).dtype

    def _prepare_observation(
        self,
        observation: Observation,
    ) -> Observation:
        observation = observation.to(
            device=self.model_device,
            dtype=self.model_dtype,
        )

        if self.normalizer is None:
            return observation

        normalized_state = self.normalizer.normalize_state(observation.state)

        return replace(
            observation,
            state=normalized_state,
        )

    def compute_loss(
        self,
        observation: Observation,
        actions: Actions,
        *,
        noise: Tensor | None = None,
        timestep: Tensor | None = None,
    ) -> Tensor:
        """Return official-style loss with shape [B, action_horizon]."""

        observation = self._prepare_observation(observation)

        actions = actions.to(
            device=self.model_device,
            dtype=self.model_dtype,
        )

        if self.normalizer is not None:
            actions = self.normalizer.normalize_actions(actions)

        (
            prefix_tokens,
            prefix_pad_masks,
            prefix_att_masks,
        ) = self.prefix_embedding(observation)

        element_loss = self.core.training_loss(
            prefix_tokens=prefix_tokens,
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
            state=observation.state,
            actions=actions,
            noise=noise,
            timestep=timestep,
        )

        # 官方实现只在 action_dim 上先取均值。
        return element_loss.to(torch.float32).mean(dim=-1)

    @torch.no_grad()
    def sample_actions(
        self,
        observation: Observation,
        *,
        num_steps: int = 10,
        noise: Tensor | None = None,
    ) -> Actions:
        """Integrate the flow field from noise at t=1 to actions at t=0."""

        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        observation = self._prepare_observation(observation)

        (
            prefix_tokens,
            prefix_pad_masks,
            prefix_att_masks,
        ) = self.prefix_embedding(observation)

        prefix_cache = self.core.prefill_prefix(
            prefix_tokens=prefix_tokens,
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
            state=observation.state,
        )

        action_shape = (
            observation.batch_size,
            self.config.action_horizon,
            self.config.action_dim,
        )

        if noise is None:
            actions = torch.randn(
                action_shape,
                device=self.model_device,
                dtype=self.model_dtype,
            )
        else:
            if noise.shape != action_shape:
                raise ValueError(f"noise must have shape {action_shape}, got {tuple(noise.shape)}")

            if not noise.is_floating_point():
                raise TypeError(f"noise must be floating point, got {noise.dtype}")

            actions = noise.to(
                device=self.model_device,
                dtype=self.model_dtype,
            )

        step_size = -1.0 / num_steps

        for step in range(num_steps):
            time_value = 1.0 + step * step_size

            timestep = torch.full(
                (observation.batch_size,),
                fill_value=time_value,
                dtype=torch.float32,
                device=self.model_device,
            )

            velocity = self.core.predict_velocity_with_cache(
                prefix_cache=prefix_cache,
                prefix_pad_masks=prefix_pad_masks,
                state=observation.state,
                noisy_actions=actions,
                timestep=timestep,
            )

            actions = actions + step_size * velocity

        # 机器人接口和反归一化通常使用 FP32。
        actions = actions.to(torch.float32)

        if self.normalizer is not None:
            actions = self.normalizer.unnormalize_actions(actions)

        return actions
