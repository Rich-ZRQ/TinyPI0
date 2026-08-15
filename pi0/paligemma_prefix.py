"""Frozen PaliGemma 2 prefix encoder."""

from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor, nn
from transformers import PaliGemmaConfig, SiglipVisionModel
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaMultiModalProjector,
)


class PaliGemmaPrefixEncoder(nn.Module):
    """Frozen SigLIP, multimodal projector, and Gemma token embedding."""

    def __init__(
        self,
        snapshot_path: str | Path,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()

        self.snapshot_path = Path(snapshot_path)
        self.checkpoint_path = self.snapshot_path / "model-00001-of-00002.safetensors"

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint shard not found: {self.checkpoint_path}")

        self.config = PaliGemmaConfig.from_pretrained(
            self.snapshot_path,
            local_files_only=True,
        )

        # 在 meta device 上创建空结构，不生成或分配随机权重。
        with torch.device("meta"):
            self.vision_tower = SiglipVisionModel(self.config.vision_config)
            self.multi_modal_projector = PaliGemmaMultiModalProjector(self.config)
            self.token_embedding = nn.Embedding(
                num_embeddings=self.config.text_config.vocab_size,
                embedding_dim=self.config.text_config.hidden_size,
            )

        self._load_module(
            module=self.vision_tower,
            checkpoint_prefix="vision_tower.",
            device=device,
            dtype=dtype,
        )
        self._load_module(
            module=self.multi_modal_projector,
            checkpoint_prefix="multi_modal_projector.",
            device=device,
            dtype=dtype,
        )
        self._load_module(
            module=self.token_embedding,
            checkpoint_prefix="language_model.model.embed_tokens.",
            device=device,
            dtype=dtype,
        )
        self._materialize_meta_buffers(self.vision_tower, device=device)

        self.requires_grad_(False)
        self.eval()

    def _load_module(
        self,
        module: nn.Module,
        checkpoint_prefix: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Selectively assign only one module's tensors."""

        state_dict: dict[str, Tensor] = {}

        with safe_open(
            self.checkpoint_path,
            framework="pt",
            device="cpu",
        ) as checkpoint:
            checkpoint_keys = set(checkpoint.keys())

            for target_name in module.state_dict():
                source_name = checkpoint_prefix + target_name

                if source_name not in checkpoint_keys:
                    raise KeyError(f"Missing checkpoint tensor: {source_name}")

                tensor = checkpoint.get_tensor(source_name)
                state_dict[target_name] = tensor.to(
                    device=device,
                    dtype=dtype,
                )

        module.load_state_dict(
            state_dict,
            strict=True,
            assign=True,
        )

    @staticmethod
    def _materialize_meta_buffers(module: nn.Module, *, device: torch.device) -> None:
        """Create non-persistent buffers omitted by safetensors checkpoints."""

        for child in module.modules():
            for name, buffer in child.named_buffers(recurse=False):
                if buffer.device.type != "meta":
                    continue

                if name != "position_ids":
                    raise RuntimeError(f"Unsupported meta buffer {type(child).__name__}.{name}")

                position_ids = torch.arange(
                    buffer.shape[-1],
                    dtype=buffer.dtype,
                    device=device,
                ).expand(buffer.shape)
                setattr(child, name, position_ids)

    @torch.no_grad()
    def encode_images(self, pixel_values: Tensor) -> Tensor:
        """Convert images [B, 3, 224, 224] into [B, 256, 2304]."""

        reference_parameter = next(self.vision_tower.parameters())
        pixel_values = pixel_values.to(
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )
        vision_output = self.vision_tower(pixel_values=pixel_values).last_hidden_state

        image_tokens = self.multi_modal_projector(vision_output)

        return image_tokens

    @torch.no_grad()
    def embed_text(self, input_ids: Tensor) -> Tensor:
        """Convert token IDs [B, L] into embeddings [B, L, 2304]."""

        text_tokens = self.token_embedding(input_ids)

        # 对齐 Gemma 2 decoder 输入端的 embedding 缩放。
        scale = self.config.text_config.hidden_size**0.5
        return text_tokens * scale

    def train(self, mode: bool = True) -> "PaliGemmaPrefixEncoder":
        """Keep the pretrained prefix modules permanently frozen."""

        super().train(False)
        return self
