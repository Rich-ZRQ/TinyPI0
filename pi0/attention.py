"""Grouped-query self-attention used by Gemma experts."""

import torch
from torch import Tensor, nn

from configs.schema import TransformerConfig
from pi0.rope import (
    GemmaRotaryEmbedding,
    apply_rotary_pos_emb,
)


def repeat_kv(
    hidden_states: Tensor,
    num_repeats: int,
) -> Tensor:
    """Repeat KV heads to match the number of query heads.

    Args:
        hidden_states:
            Tensor [B, K, S, D].

        num_repeats:
            Number of query-head groups sharing each KV head.

    Returns:
        Tensor [B, K * num_repeats, S, D].
    """

    if hidden_states.ndim != 4:
        raise ValueError(f"hidden_states must have shape [B, K, S, D], got {tuple(hidden_states.shape)}")

    if num_repeats <= 0:
        raise ValueError(f"num_repeats must be positive, got {num_repeats}")

    if num_repeats == 1:
        return hidden_states

    batch_size, num_kv_heads, sequence_length, head_dim = hidden_states.shape

    expanded = hidden_states[:, :, None, :, :].expand(
        batch_size,
        num_kv_heads,
        num_repeats,
        sequence_length,
        head_dim,
    )

    return expanded.reshape(
        batch_size,
        num_kv_heads * num_repeats,
        sequence_length,
        head_dim,
    )


class GemmaAttention(nn.Module):
    """Gemma grouped-query self-attention."""

    def __init__(
        self,
        config: TransformerConfig,
        *,
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()

        self.config = config
        self.width = config.width
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim

        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.width,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.width,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.width,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.width,
            bias=False,
        )

        self.rotary_embedding = GemmaRotaryEmbedding(
            head_dim=self.head_dim,
            base=rope_base,
        )

    def project_qkv(
        self,
        hidden_states: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Project hidden states into separated Q/K/V heads."""

        self._validate_hidden_states(hidden_states)

        batch_size, sequence_length, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        key = key.reshape(
            batch_size,
            sequence_length,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        value = value.reshape(
            batch_size,
            sequence_length,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        return query, key, value

    def attend(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run masked scaled dot-product attention."""

        key = repeat_kv(
            key,
            self.num_kv_groups,
        )
        value = repeat_kv(
            value,
            self.num_kv_groups,
        )

        attention_scores = torch.matmul(
            query.to(torch.float32),
            key.to(torch.float32).transpose(-2, -1),
        )
        attention_scores = attention_scores * self.scaling

        expected_mask_shape = (
            query.shape[0],
            query.shape[2],
            key.shape[2],
        )

        if attention_mask.shape != expected_mask_shape:
            raise ValueError(f"attention_mask must have shape {expected_mask_shape}, got {tuple(attention_mask.shape)}")

        if attention_mask.dtype != torch.bool:
            raise TypeError(f"attention_mask must be bool, got {attention_mask.dtype}")

        if attention_mask.device != query.device:
            raise ValueError("attention_mask and query must be on the same device")

        attention_scores = attention_scores.masked_fill(
            ~attention_mask[:, None, :, :],
            torch.finfo(attention_scores.dtype).min,
        )

        attention_probabilities = torch.softmax(
            attention_scores,
            dim=-1,
            dtype=torch.float32,
        ).to(dtype=query.dtype)

        attention_output = torch.matmul(
            attention_probabilities,
            value,
        )

        return (
            attention_output,
            attention_probabilities,
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply QKV projections, RoPE, GQA and output projection."""

        query, key, value = self.project_qkv(hidden_states)

        reference = query[:, 0]
        cosine, sine = self.rotary_embedding(
            reference,
            position_ids,
        )

        query, key = apply_rotary_pos_emb(
            query,
            key,
            cosine,
            sine,
        )

        attention_output, attention_probabilities = self.attend(
            query,
            key,
            value,
            attention_mask,
        )

        batch_size, _, sequence_length, _ = attention_output.shape

        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .reshape(
                batch_size,
                sequence_length,
                self.num_heads * self.head_dim,
            )
        )

        output = self.o_proj(attention_output)

        return output, attention_probabilities

    def _validate_hidden_states(
        self,
        hidden_states: Tensor,
    ) -> None:
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must have shape [B, S, width], got {tuple(hidden_states.shape)}")

        if hidden_states.shape[-1] != self.width:
            raise ValueError(f"expected hidden width {self.width}, got {hidden_states.shape[-1]}")

        if not hidden_states.is_floating_point():
            raise TypeError(f"hidden_states must be floating point, got {hidden_states.dtype}")
