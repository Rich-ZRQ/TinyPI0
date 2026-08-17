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
        *,
        trainable_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.config = config

        parameter_dtype = (
            trainable_dtype
            or {
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }[config.dtype]
        )

        if parameter_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"trainable_dtype must be float32 or bfloat16, got {parameter_dtype}")

        self.prefix_embedding = Pi0PrefixEmbedding(
            config=config,
            prefix_encoder=prefix_encoder,
            projection_dtype=parameter_dtype,
        )

        reference_parameter = next(prefix_encoder.parameters())

        self.core = Pi0Core(config).to(
            device=reference_parameter.device,
            dtype=parameter_dtype,
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
        action_dim_mask: Tensor | None = None,
    ) -> Tensor:
        """Return official-style loss with shape [B, action_horizon]."""

        observation = self._prepare_observation(observation)

        actions = actions.to(
            device=self.model_device,
            dtype=self.model_dtype,
        )

        if self.normalizer is not None:
            actions = self.normalizer.normalize_actions(actions)

        validated_dim_mask = self._prepare_action_dim_mask(
            action_dim_mask,
            batch_size=observation.batch_size,
        )

        if validated_dim_mask is not None:
            numeric_dim_mask = validated_dim_mask[:, None, :].to(dtype=actions.dtype)
            actions = actions * numeric_dim_mask

            if noise is None:
                noise = torch.randn_like(actions)
            else:
                noise = noise.to(
                    device=self.model_device,
                    dtype=self.model_dtype,
                )

            noise = noise * numeric_dim_mask

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

        element_loss = element_loss.to(torch.float32)

        if validated_dim_mask is None:
            return element_loss.mean(dim=-1)

        dimension_mask = validated_dim_mask.to(
            device=element_loss.device,
            dtype=element_loss.dtype,
        )[:, None, :]
        return (element_loss * dimension_mask).sum(dim=-1) / dimension_mask.sum(dim=-1).clamp_min(1)

    @torch.no_grad()
    def sample_actions(
        self,
        observation: Observation,
        *,
        num_steps: int = 10,
        noise: Tensor | None = None,
        action_dim_mask: Tensor | None = None,
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
        validated_dim_mask = self._prepare_action_dim_mask(
            action_dim_mask,
            batch_size=observation.batch_size,
            infer_from_normalizer=True,
        )
        numeric_dim_mask = (
            None
            if validated_dim_mask is None
            else validated_dim_mask[:, None, :].to(
                device=self.model_device,
                dtype=self.model_dtype,
            )
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

        if numeric_dim_mask is not None:
            actions = actions * numeric_dim_mask

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

            if numeric_dim_mask is not None:
                velocity = velocity * numeric_dim_mask

            actions = actions + step_size * velocity

        # 机器人接口和反归一化通常使用 FP32。
        actions = actions.to(torch.float32)

        if self.normalizer is not None:
            actions = self.normalizer.unnormalize_actions(actions)

        return actions

    def _prepare_action_dim_mask(
        self,
        action_dim_mask: Tensor | None,
        *,
        batch_size: int,
        infer_from_normalizer: bool = False,
    ) -> Tensor | None:
        if action_dim_mask is None and infer_from_normalizer and self.normalizer is not None:
            robot_dim = min(
                self.config.action_dim,
                self.normalizer.action_mean.shape[0],
            )
            action_dim_mask = (
                torch.arange(
                    self.config.action_dim,
                    device=self.model_device,
                )[None, :].expand(batch_size, -1)
                < robot_dim
            )

        if action_dim_mask is None:
            return None

        expected_shape = (
            batch_size,
            self.config.action_dim,
        )

        if action_dim_mask.shape != expected_shape:
            raise ValueError(f"action_dim_mask must have shape {expected_shape}, got {tuple(action_dim_mask.shape)}")
        if action_dim_mask.dtype != torch.bool:
            raise TypeError(f"action_dim_mask must be bool, got {action_dim_mask.dtype}")
        if not torch.all(action_dim_mask.any(dim=1)):
            raise ValueError("action_dim_mask must enable at least one dimension per batch item")

        return action_dim_mask.to(device=self.model_device)
