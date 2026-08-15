"""A single pre-normalized Gemma decoder layer."""

from torch import Tensor, nn

from configs.schema import TransformerConfig
from pi0.attention import GemmaAttention
from pi0.mlp import GemmaMLP
from pi0.rms_norm import GemmaRMSNorm


class GemmaDecoderLayer(nn.Module):
    """One Gemma transformer layer."""

    def __init__(
        self,
        config: TransformerConfig,
        *,
        rms_norm_eps: float = 1e-6,
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()

        self.width = config.width

        self.self_attn = GemmaAttention(
            config,
            rope_base=rope_base,
        )
        self.mlp = GemmaMLP(config)

        self.input_layernorm = GemmaRMSNorm(
            config.width,
            eps=rms_norm_eps,
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.width,
            eps=rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run attention and MLP residual blocks."""

        self._validate_hidden_states(hidden_states)

        residual = hidden_states

        normalized = self.input_layernorm(hidden_states)

        attention_output, attention_probabilities = self.self_attn(
            normalized,
            position_ids,
            attention_mask,
        )

        hidden_states = residual + attention_output

        residual = hidden_states

        normalized = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normalized)

        hidden_states = residual + mlp_output

        return hidden_states, attention_probabilities

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
