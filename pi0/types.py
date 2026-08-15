"""Canonical PyTorch tensor contracts used by pi0."""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

from configs.schema import Pi0Config

IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


Actions = Tensor  # 别名


@dataclass(frozen=True)
class Observation:
    """A batch of canonical, model-ready pi0 observations."""

    images: dict[str, Tensor]
    image_masks: dict[str, Tensor]
    state: Tensor
    tokenized_prompt: Tensor | None = None
    tokenized_prompt_mask: Tensor | None = None

    @property
    def batch_size(self) -> int:
        if self.state.ndim < 1:
            raise ValueError("state must have a batch dimension")
        return self.state.shape[0]

    @property
    def device(self) -> torch.device:
        return self.state.device

    def validate(self, config: Pi0Config) -> None:
        self._validate_state(config)
        self._validate_images(config)
        self._validate_prompt(config)

    def _validate_state(self, config: Pi0Config) -> None:
        expected_shape = (self.batch_size, config.action_dim)

        if self.state.shape != expected_shape:
            raise ValueError(f"state must have shape {expected_shape}, got {tuple(self.state.shape)}")

        if not self.state.is_floating_point():
            raise TypeError(f"state must be floating point, got {self.state.dtype}")

    def _validate_images(self, config: Pi0Config) -> None:
        if set(self.images) != set(IMAGE_KEYS):
            raise ValueError(f"images must contain exactly {IMAGE_KEYS}, got {tuple(self.images)}")

        if set(self.image_masks) != set(IMAGE_KEYS):
            raise ValueError(f"image_masks must contain exactly {IMAGE_KEYS}, got {tuple(self.image_masks)}")

        expected_image_shape = (
            self.batch_size,
            config.vision.num_channels,
            config.vision.image_size,
            config.vision.image_size,
        )  # (B, C, H, W)
        expected_mask_shape = (self.batch_size,)

        for key in IMAGE_KEYS:
            image = self.images[key]
            mask = self.image_masks[key]

            if image.shape != expected_image_shape:
                raise ValueError(f"image {key!r} must have shape {expected_image_shape}, got {tuple(image.shape)}")

            if not image.is_floating_point():
                raise TypeError(f"image {key!r} must be floating point, got {image.dtype}")

            if mask.shape != expected_mask_shape:
                raise ValueError(f"image mask {key!r} must have shape {expected_mask_shape}, got {tuple(mask.shape)}")

            if mask.dtype != torch.bool:
                raise TypeError(f"image mask {key!r} must be bool, got {mask.dtype}")

            if image.device != self.state.device:
                raise ValueError(f"image {key!r} and state must be on the same device")

            if mask.device != self.state.device:
                raise ValueError(f"image mask {key!r} and state must be on the same device")

    def _validate_prompt(self, config: Pi0Config) -> None:
        has_tokens = self.tokenized_prompt is not None
        has_mask = self.tokenized_prompt_mask is not None

        if has_tokens != has_mask:
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together")

        if not has_tokens:
            return

        assert self.tokenized_prompt is not None
        assert self.tokenized_prompt_mask is not None

        expected_shape = (
            self.batch_size,
            config.max_token_len,
        )

        if self.tokenized_prompt.shape != expected_shape:
            raise ValueError(
                f"tokenized_prompt must have shape {expected_shape}, got {tuple(self.tokenized_prompt.shape)}"
            )

        if self.tokenized_prompt_mask.shape != expected_shape:
            raise ValueError(
                f"tokenized_prompt_mask must have shape {expected_shape}, got {tuple(self.tokenized_prompt_mask.shape)}"
            )

        if self.tokenized_prompt.dtype != torch.long:
            raise TypeError(f"tokenized_prompt must use torch.long, got {self.tokenized_prompt.dtype}")

        if self.tokenized_prompt_mask.dtype != torch.bool:
            raise TypeError(f"tokenized_prompt_mask must use torch.bool, got {self.tokenized_prompt_mask.dtype}")

        if self.tokenized_prompt.device != self.state.device:
            raise ValueError("tokenized_prompt and state must be on the same device")

        if self.tokenized_prompt_mask.device != self.state.device:
            raise ValueError("tokenized_prompt_mask and state must be on the same device")

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> Self:
        target_dtype = dtype or self.state.dtype

        return type(self)(
            images={key: image.to(device=device, dtype=target_dtype) for key, image in self.images.items()},
            image_masks={key: mask.to(device=device) for key, mask in self.image_masks.items()},
            state=self.state.to(
                device=device,
                dtype=target_dtype,
            ),
            tokenized_prompt=(None if self.tokenized_prompt is None else self.tokenized_prompt.to(device=device)),
            tokenized_prompt_mask=(
                None if self.tokenized_prompt_mask is None else self.tokenized_prompt_mask.to(device=device)
            ),
        )


def validate_actions(
    actions: Actions,
    *,
    config: Pi0Config,
    batch_size: int,
) -> None:
    expected_shape = (
        batch_size,
        config.action_horizon,
        config.action_dim,
    )

    if actions.shape != expected_shape:
        raise ValueError(f"actions must have shape {expected_shape}, got {tuple(actions.shape)}")

    if not actions.is_floating_point():
        raise TypeError(f"actions must be floating point, got {actions.dtype}")
