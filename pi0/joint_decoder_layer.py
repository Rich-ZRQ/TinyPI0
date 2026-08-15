"""Joint decoder layer connecting PaliGemma and the action expert."""

import torch
from torch import Tensor, nn

from configs.schema import Pi0Config
from pi0.decoder_layer import GemmaDecoderLayer
from pi0.rope import apply_rotary_pos_emb


class JointDecoderLayer(nn.Module):
    """One pi0 layer with joint attention across two experts."""

    def __init__(
        self,
        config: Pi0Config,
        *,
        rms_norm_eps: float = 1e-6,
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()

        self.paligemma_layer = GemmaDecoderLayer(
            config.paligemma,
            rms_norm_eps=rms_norm_eps,
            rope_base=rope_base,
        )
        self.action_expert_layer = GemmaDecoderLayer(
            config.action_expert,
            rms_norm_eps=rms_norm_eps,
            rope_base=rope_base,
        )

        self.paligemma_width = config.paligemma.width
        self.action_expert_width = config.action_expert.width

    def forward(
        self,
        paligemma_hidden_states: Tensor,
        action_hidden_states: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run one joint-attention dual-expert layer."""

        self._validate_inputs(
            paligemma_hidden_states,
            action_hidden_states,
            position_ids,
            attention_mask,
        )

        paligemma_residual = paligemma_hidden_states
        action_residual = action_hidden_states

        paligemma_normalized = self.paligemma_layer.input_layernorm(paligemma_hidden_states)
        action_normalized = self.action_expert_layer.input_layernorm(action_hidden_states)

        paligemma_query, paligemma_key, paligemma_value = self.paligemma_layer.self_attn.project_qkv(
            paligemma_normalized
        )

        action_query, action_key, action_value = self.action_expert_layer.self_attn.project_qkv(action_normalized)

        joint_query = torch.cat(
            [paligemma_query, action_query],
            dim=2,
        )
        joint_key = torch.cat(
            [paligemma_key, action_key],
            dim=2,
        )
        joint_value = torch.cat(
            [paligemma_value, action_value],
            dim=2,
        )

        rotary_reference = joint_query[:, 0]

        cosine, sine = self.paligemma_layer.self_attn.rotary_embedding(
            rotary_reference,
            position_ids,
        )

        joint_query, joint_key = apply_rotary_pos_emb(
            joint_query,
            joint_key,
            cosine,
            sine,
        )

        joint_attention_output, attention_probabilities = self.paligemma_layer.self_attn.attend(
            joint_query,
            joint_key,
            joint_value,
            attention_mask,
        )

        joint_attention_output = (
            joint_attention_output.transpose(1, 2)
            .contiguous()
            .reshape(
                joint_attention_output.shape[0],
                joint_attention_output.shape[2],
                -1,
            )
        )

        paligemma_length = paligemma_hidden_states.shape[1]
        action_length = action_hidden_states.shape[1]

        paligemma_attention_output, action_attention_output = joint_attention_output.split(
            [paligemma_length, action_length],
            dim=1,
        )

        paligemma_attention_output = self.paligemma_layer.self_attn.o_proj(paligemma_attention_output)
        action_attention_output = self.action_expert_layer.self_attn.o_proj(action_attention_output)

        paligemma_hidden_states = paligemma_residual + paligemma_attention_output
        action_hidden_states = action_residual + action_attention_output

        paligemma_residual = paligemma_hidden_states
        action_residual = action_hidden_states

        paligemma_normalized = self.paligemma_layer.post_attention_layernorm(paligemma_hidden_states)
        action_normalized = self.action_expert_layer.post_attention_layernorm(action_hidden_states)

        paligemma_mlp_output = self.paligemma_layer.mlp(paligemma_normalized)
        action_mlp_output = self.action_expert_layer.mlp(action_normalized)

        paligemma_hidden_states = paligemma_residual + paligemma_mlp_output
        action_hidden_states = action_residual + action_mlp_output

        return (
            paligemma_hidden_states,
            action_hidden_states,
            attention_probabilities,
        )

    def encode_prefix(
        self,
        prefix_hidden_states: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the prefix expert and return its rotated K/V cache."""

        self._validate_stream(
            "prefix_hidden_states",
            prefix_hidden_states,
            self.paligemma_width,
        )

        residual = prefix_hidden_states

        normalized = self.paligemma_layer.input_layernorm(prefix_hidden_states)

        query, key, value = self.paligemma_layer.self_attn.project_qkv(normalized)

        rotary_reference = query[:, 0]

        cosine, sine = self.paligemma_layer.self_attn.rotary_embedding(
            rotary_reference,
            position_ids,
        )

        query, rotated_key = apply_rotary_pos_emb(
            query,
            key,
            cosine,
            sine,
        )

        attention_output, _ = self.paligemma_layer.self_attn.attend(
            query,
            rotated_key,
            value,
            attention_mask,
        )

        attention_output = self._merge_heads(attention_output)

        attention_output = self.paligemma_layer.self_attn.o_proj(attention_output)

        prefix_hidden_states = residual + attention_output

        residual = prefix_hidden_states

        normalized = self.paligemma_layer.post_attention_layernorm(prefix_hidden_states)

        mlp_output = self.paligemma_layer.mlp(normalized)

        prefix_hidden_states = residual + mlp_output

        # key已经应用RoPE，value不需要应用RoPE。
        return (
            prefix_hidden_states,
            rotated_key,
            value,
        )

    def decode_suffix(
        self,
        suffix_hidden_states: Tensor,
        prefix_key: Tensor,
        prefix_value: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Run action tokens against cached prefix K/V."""

        self._validate_stream(
            "suffix_hidden_states",
            suffix_hidden_states,
            self.action_expert_width,
        )

        if prefix_key.shape != prefix_value.shape:
            raise ValueError(
                f"prefix_key and prefix_value must have the same shape, got {prefix_key.shape} and {prefix_value.shape}"
            )

        if prefix_key.ndim != 4:
            raise ValueError("prefix cache must have shape [B, num_kv_heads, prefix_length, head_dim]")

        if prefix_key.shape[0] != suffix_hidden_states.shape[0]:
            raise ValueError("prefix cache and suffix must have the same batch size")

        if prefix_key.device != suffix_hidden_states.device:
            raise ValueError("prefix cache and suffix must be on the same device")

        if prefix_key.dtype != suffix_hidden_states.dtype:
            raise TypeError("prefix cache and suffix must have the same dtype")

        residual = suffix_hidden_states

        normalized = self.action_expert_layer.input_layernorm(suffix_hidden_states)

        query, suffix_key, suffix_value = self.action_expert_layer.self_attn.project_qkv(normalized)

        rotary_reference = query[:, 0]

        cosine, sine = self.action_expert_layer.self_attn.rotary_embedding(
            rotary_reference,
            position_ids,
        )

        query, suffix_key = apply_rotary_pos_emb(
            query,
            suffix_key,
            cosine,
            sine,
        )

        joint_key = torch.cat(
            [prefix_key, suffix_key],
            dim=2,
        )
        joint_value = torch.cat(
            [prefix_value, suffix_value],
            dim=2,
        )

        attention_output, _ = self.action_expert_layer.self_attn.attend(
            query,
            joint_key,
            joint_value,
            attention_mask,
        )

        attention_output = self._merge_heads(attention_output)

        attention_output = self.action_expert_layer.self_attn.o_proj(attention_output)

        suffix_hidden_states = residual + attention_output

        residual = suffix_hidden_states

        normalized = self.action_expert_layer.post_attention_layernorm(suffix_hidden_states)

        mlp_output = self.action_expert_layer.mlp(normalized)

        return residual + mlp_output

    @staticmethod
    def _merge_heads(
        attention_output: Tensor,
    ) -> Tensor:
        """Convert [B, H, S, D] into [B, S, H*D]."""

        return (
            attention_output.transpose(1, 2)
            .contiguous()
            .reshape(
                attention_output.shape[0],
                attention_output.shape[2],
                -1,
            )
        )

    def _validate_inputs(
        self,
        paligemma_hidden_states: Tensor,
        action_hidden_states: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
    ) -> None:
        self._validate_stream(
            "paligemma_hidden_states",
            paligemma_hidden_states,
            self.paligemma_width,
        )
        self._validate_stream(
            "action_hidden_states",
            action_hidden_states,
            self.action_expert_width,
        )

        if paligemma_hidden_states.shape[0] != action_hidden_states.shape[0]:
            raise ValueError("both expert streams must have the same batch size")

        if paligemma_hidden_states.device != action_hidden_states.device:
            raise ValueError("both expert streams must be on the same device")

        if paligemma_hidden_states.dtype != action_hidden_states.dtype:
            raise TypeError("both expert streams must have the same dtype")

        batch_size = paligemma_hidden_states.shape[0]
        total_length = paligemma_hidden_states.shape[1] + action_hidden_states.shape[1]

        expected_position_shape = (
            batch_size,
            total_length,
        )
        expected_mask_shape = (
            batch_size,
            total_length,
            total_length,
        )

        if position_ids.shape != expected_position_shape:
            raise ValueError(f"position_ids must have shape {expected_position_shape}, got {tuple(position_ids.shape)}")

        if position_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError("position_ids must use an integer dtype")

        if attention_mask.shape != expected_mask_shape:
            raise ValueError(f"attention_mask must have shape {expected_mask_shape}, got {tuple(attention_mask.shape)}")

        if attention_mask.dtype != torch.bool:
            raise TypeError(f"attention_mask must be bool, got {attention_mask.dtype}")

        if position_ids.device != paligemma_hidden_states.device:
            raise ValueError("position_ids and hidden states must be on the same device")

        if attention_mask.device != paligemma_hidden_states.device:
            raise ValueError("attention_mask and hidden states must be on the same device")

    @staticmethod
    def _validate_stream(
        name: str,
        hidden_states: Tensor,
        expected_width: int,
    ) -> None:
        if hidden_states.ndim != 3:
            raise ValueError(f"{name} must have shape [B, S, width], got {tuple(hidden_states.shape)}")

        if hidden_states.shape[-1] != expected_width:
            raise ValueError(f"{name} expected width {expected_width}, got {hidden_states.shape[-1]}")

        if not hidden_states.is_floating_point():
            raise TypeError(f"{name} must be floating point, got {hidden_states.dtype}")
