"""SO-ARM101 profiles selected from RTX 3050 Ti inference benchmarks."""

from configs.schema import Pi0Config, RuntimeConfig, TransformerConfig, VisionConfig


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


# Recommended starting point for 71 episodes / 54,699 frames.
# It keeps the official Gemma-style 8x prefix MLP and 4x action MLP ratios.
SO101_RECOMMENDED = Pi0Config(
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


# Larger ablation profile. Use it only if held-out episode validation improves.
SO101_LARGE = Pi0Config(
    vision=_vision_config(1536),
    paligemma=TransformerConfig(
        width=1536,
        depth=8,
        mlp_dim=12_288,
        num_heads=12,
        num_kv_heads=1,
        head_dim=128,
    ),
    action_expert=TransformerConfig(
        width=768,
        depth=8,
        mlp_dim=3072,
        num_heads=12,
        num_kv_heads=1,
        head_dim=128,
    ),
    action_dim=32,
    action_horizon=50,
    max_token_len=48,
    dtype="bfloat16",
)


SO101_LOCAL_RUNTIME = RuntimeConfig(
    batch_size=1,
    gradient_checkpointing=False,
    compile_model=False,
    num_workers=0,
)

SO101_SERVER_RUNTIME = RuntimeConfig(
    batch_size=8,
    gradient_checkpointing=True,
    compile_model=True,
    num_workers=4,
)
