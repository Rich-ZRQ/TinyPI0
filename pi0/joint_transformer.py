"""Stacked dual-expert transformer backbone used by pi0."""

from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from configs.schema import Pi0Config
from pi0.joint_decoder_layer import JointDecoderLayer
from pi0.rms_norm import GemmaRMSNorm

PrefixKVCache = tuple[tuple[Tensor, Tensor], ...]


class JointTransformer(nn.Module):
    """Stack joint decoder layers and normalize both outputs."""

    def __init__(
        self,
        config: Pi0Config,
        *,
        rms_norm_eps: float = 1e-6,
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()

        self.config = config
        self.depth = config.paligemma.depth
        self.gradient_checkpointing = False

        self.layers = nn.ModuleList(
            [
                JointDecoderLayer(
                    config,
                    rms_norm_eps=rms_norm_eps,
                    rope_base=rope_base,
                )
                for _ in range(self.depth)
            ]
        )

        self.paligemma_norm = GemmaRMSNorm(
            config.paligemma.width,
            eps=rms_norm_eps,
        )
        self.action_expert_norm = GemmaRMSNorm(
            config.action_expert.width,
            eps=rms_norm_eps,
        )

    def forward(
        self,
        paligemma_hidden_states: Tensor,
        action_hidden_states: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run all joint layers followed by final RMSNorm."""

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:

                def layer_forward(
                    prefix: Tensor,
                    suffix: Tensor,
                    positions: Tensor,
                    mask: Tensor,
                    current_layer: JointDecoderLayer = layer,
                ) -> tuple[Tensor, Tensor]:
                    prefix, suffix, _ = current_layer(prefix, suffix, positions, mask)
                    return prefix, suffix

                paligemma_hidden_states, action_hidden_states = checkpoint(
                    layer_forward,
                    paligemma_hidden_states,
                    action_hidden_states,
                    position_ids,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                (
                    paligemma_hidden_states,
                    action_hidden_states,
                    _,
                ) = layer(
                    paligemma_hidden_states,
                    action_hidden_states,
                    position_ids,
                    attention_mask,
                )

        paligemma_hidden_states = self.paligemma_norm(paligemma_hidden_states)
        action_hidden_states = self.action_expert_norm(action_hidden_states)

        return (
            paligemma_hidden_states,
            action_hidden_states,
        )

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Trade extra forward compute for lower training activation memory."""

        self.gradient_checkpointing = enabled

    def prefill_prefix(
        self,
        prefix_hidden_states: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, PrefixKVCache]:
        """Run the prefix once and build one K/V cache per layer."""

        cache_entries: list[tuple[Tensor, Tensor]] = []

        for layer in self.layers:
            (
                prefix_hidden_states,
                prefix_key,
                prefix_value,
            ) = layer.encode_prefix(
                prefix_hidden_states=prefix_hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

            cache_entries.append(
                (
                    prefix_key,
                    prefix_value,
                )
            )

        prefix_hidden_states = self.paligemma_norm(prefix_hidden_states)

        return (
            prefix_hidden_states,
            tuple(cache_entries),
        )

    def forward_suffix_with_cache(
        self,
        suffix_hidden_states: Tensor,
        prefix_cache: PrefixKVCache,
        position_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Run only the action expert using cached prefix K/V."""

        if len(prefix_cache) != self.depth:
            raise ValueError(
                f"prefix cache must contain one entry per layer, expected {self.depth}, got {len(prefix_cache)}"
            )

        for layer, cache_entry in zip(
            self.layers,
            prefix_cache,
            strict=True,
        ):
            prefix_key, prefix_value = cache_entry

            suffix_hidden_states = layer.decode_suffix(
                suffix_hidden_states=suffix_hidden_states,
                prefix_key=prefix_key,
                prefix_value=prefix_value,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

        return self.action_expert_norm(suffix_hidden_states)
