"""Build Tiny pi0 prefix tokens from frozen PaliGemma features."""

import torch
from torch import Tensor, nn

from configs.schema import Pi0Config
from pi0.paligemma_prefix import PaliGemmaPrefixEncoder
from pi0.types import IMAGE_KEYS, Observation


class Pi0PrefixEmbedding(nn.Module):
    """Assemble image and language tokens for the prefix expert."""

    def __init__(
        self,
        config: Pi0Config,
        prefix_encoder: PaliGemmaPrefixEncoder,
        *,
        projection_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.prefix_encoder = prefix_encoder

        source_width = prefix_encoder.config.text_config.hidden_size
        target_width = config.paligemma.width

        reference_parameter = next(prefix_encoder.parameters())
        target_dtype = (
            projection_dtype
            or {
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }[config.dtype]
        )

        # 图像和文本共享一个映射，以尽量保留它们已经对齐的特征空间。
        self.input_projection = nn.Linear(
            source_width,
            target_width,
            bias=False,
            device=reference_parameter.device,
            dtype=target_dtype,
        )

    def forward(
        self,
        observation: Observation,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return prefix tokens, padding masks and block masks."""

        observation.validate(self.config)

        token_parts: list[Tensor] = []
        pad_mask_parts: list[Tensor] = []
        att_mask_parts: list[Tensor] = []

        for key in IMAGE_KEYS:
            image_mask = observation.image_masks[key]

            # 整个 batch 都不存在的相机不执行 SigLIP，
            # 也不把无效 token 放进 Transformer。
            if not bool(torch.any(image_mask)):
                continue

            image_features = self.prefix_encoder.encode_images(observation.images[key])

            image_tokens = self.input_projection(image_features.to(dtype=self.input_projection.weight.dtype))

            # batch 中某个样本没有该相机时，将其 token 清零。
            image_tokens = image_tokens * image_mask[:, None, None].to(dtype=image_tokens.dtype)

            token_count = image_tokens.shape[1]

            image_token_mask = image_mask[:, None].expand(
                -1,
                token_count,
            )

            # 所有 prefix token 属于同一个双向注意力块。
            image_att_mask = torch.zeros(
                image_token_mask.shape,
                dtype=torch.bool,
                device=image_token_mask.device,
            )

            token_parts.append(image_tokens)
            pad_mask_parts.append(image_token_mask)
            att_mask_parts.append(image_att_mask)

        if observation.tokenized_prompt is not None:
            assert observation.tokenized_prompt_mask is not None

            text_features = self.prefix_encoder.embed_text(observation.tokenized_prompt)

            text_tokens = self.input_projection(text_features.to(dtype=self.input_projection.weight.dtype))

            text_att_mask = torch.zeros(
                observation.tokenized_prompt_mask.shape,
                dtype=torch.bool,
                device=observation.device,
            )

            token_parts.append(text_tokens)
            pad_mask_parts.append(observation.tokenized_prompt_mask)
            att_mask_parts.append(text_att_mask)

        if not token_parts:
            raise ValueError("At least one valid image or prompt is required")

        prefix_tokens = torch.cat(
            token_parts,
            dim=1,
        )
        prefix_pad_masks = torch.cat(
            pad_mask_parts,
            dim=1,
        )
        prefix_att_masks = torch.cat(
            att_mask_parts,
            dim=1,
        )

        return (
            prefix_tokens,
            prefix_pad_masks,
            prefix_att_masks,
        )
