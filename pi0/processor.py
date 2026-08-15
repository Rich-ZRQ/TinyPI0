"""Input preprocessing for Tiny pi0."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from transformers import AutoImageProcessor, AutoTokenizer

from configs.schema import Pi0Config
from pi0.types import IMAGE_KEYS, Observation

RawImage: TypeAlias = Image.Image | np.ndarray | Tensor


class Pi0Processor:
    """Convert raw robot inputs into a canonical Observation."""

    def __init__(
        self,
        config: Pi0Config,
        snapshot_path: str | Path,
    ) -> None:
        self.config = config
        self.snapshot_path = Path(snapshot_path)

        self.image_processor = AutoImageProcessor.from_pretrained(
            self.snapshot_path,
            local_files_only=True,
            use_fast=False,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.snapshot_path,
            local_files_only=True,
        )

        if self.tokenizer.bos_token_id is None:
            raise ValueError("Tokenizer must define bos_token_id")

        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id")

        self.newline_token_ids = self.tokenizer.encode(
            "\n",
            add_special_tokens=False,
        )

        if not self.newline_token_ids:
            raise ValueError("Tokenizer produced no token for newline")

    def __call__(
        self,
        *,
        images: Mapping[
            str,
            Sequence[RawImage | None],
        ],
        prompts: Sequence[str],
        state: Tensor | np.ndarray,
    ) -> Observation:
        """Build a CPU Observation from a batch of raw inputs."""

        batch_size = len(prompts)

        if batch_size <= 0:
            raise ValueError("prompts must contain at least one item")

        unknown_image_keys = set(images) - set(IMAGE_KEYS)

        if unknown_image_keys:
            raise ValueError(f"Unknown image keys: {sorted(unknown_image_keys)}")

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device="cpu",
        )

        expected_state_shape = (
            batch_size,
            self.config.action_dim,
        )

        if state_tensor.shape != expected_state_shape:
            raise ValueError(f"state must have shape {expected_state_shape}, got {tuple(state_tensor.shape)}")

        processed_images: dict[str, Tensor] = {}
        image_masks: dict[str, Tensor] = {}

        for key in IMAGE_KEYS:
            camera_images = images.get(key)

            if camera_images is None:
                processed_images[key] = torch.zeros(
                    batch_size,
                    self.config.vision.num_channels,
                    self.config.vision.image_size,
                    self.config.vision.image_size,
                    dtype=torch.float32,
                )
                image_masks[key] = torch.zeros(
                    batch_size,
                    dtype=torch.bool,
                )
                continue

            if len(camera_images) != batch_size:
                raise ValueError(f"Camera {key!r} must contain {batch_size} images, got {len(camera_images)}")

            valid_mask = torch.tensor(
                [image is not None for image in camera_images],
                dtype=torch.bool,
            )

            black_image = Image.new(
                mode="RGB",
                size=(
                    self.config.vision.image_size,
                    self.config.vision.image_size,
                ),
            )

            filled_images = [image if image is not None else black_image for image in camera_images]

            pixel_values = self.image_processor(
                images=filled_images,
                return_tensors="pt",
            )["pixel_values"]

            processed_images[key] = pixel_values.to(dtype=torch.float32)
            image_masks[key] = valid_mask

        (
            tokenized_prompt,
            tokenized_prompt_mask,
        ) = self.tokenize_prompts(prompts)

        observation = Observation(
            images=processed_images,
            image_masks=image_masks,
            state=state_tensor,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )

        observation.validate(self.config)
        return observation

    def tokenize_prompts(
        self,
        prompts: Sequence[str],
    ) -> tuple[Tensor, Tensor]:
        """Apply the official pi0 prompt-tokenization convention."""

        all_token_ids: list[list[int]] = []
        all_token_masks: list[list[bool]] = []

        for prompt in prompts:
            cleaned_prompt = prompt.strip().replace("_", " ").replace("\n", " ")

            prompt_ids = self.tokenizer.encode(
                cleaned_prompt,
                add_special_tokens=False,
            )

            token_ids = [
                self.tokenizer.bos_token_id,
                *prompt_ids,
                *self.newline_token_ids,
            ]

            token_ids = token_ids[: self.config.max_token_len]

            valid_length = len(token_ids)
            padding_length = self.config.max_token_len - valid_length

            token_ids.extend([self.tokenizer.pad_token_id] * padding_length)

            token_mask = [True] * valid_length + [False] * padding_length

            all_token_ids.append(token_ids)
            all_token_masks.append(token_mask)

        return (
            torch.tensor(
                all_token_ids,
                dtype=torch.long,
            ),
            torch.tensor(
                all_token_masks,
                dtype=torch.bool,
            ),
        )
