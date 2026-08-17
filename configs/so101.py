"""Tiny pi0 capacity used for the two-camera SO-ARM101 experiment."""

from configs.schema import Pi0Config, TransformerConfig, VisionConfig


def _vision_config(projection_dim: int) -> VisionConfig:
    """Describe the frozen SigLIP So400m/14 frontend contract."""

    return VisionConfig(
        image_size=224,
        patch_size=14,
        width=1152,
        depth=27,
        mlp_dim=4304,
        num_heads=16,
        projection_dim=projection_dim,
    )


# Capacity used by the current training artifacts. It keeps Gemma-style 8x
# prefix MLP and 4x action-expert MLP ratios while remaining a Tiny model.
SO101_TINY = Pi0Config(
    vision=_vision_config(1024),
    paligemma=TransformerConfig(
        width=1024,
        depth=8,
        mlp_dim=8192,
        num_heads=8,
        num_kv_heads=1,
        head_dim=128,
    ),
    action_expert=TransformerConfig(
        width=512,
        depth=8,
        mlp_dim=2048,
        num_heads=8,
        num_kv_heads=1,
        head_dim=128,
    ),
    action_dim=32,
    action_horizon=50,
    max_token_len=48,
    dtype="bfloat16",
)
