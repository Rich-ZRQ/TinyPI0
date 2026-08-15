"""Official-size pi0 profile for server execution."""

from configs.schema import (
    Pi0Config,
    RuntimeConfig,
    TransformerConfig,
    VisionConfig,
)

FULL_PI0 = Pi0Config(
    vision=VisionConfig(
        image_size=224,
        patch_size=14,
        width=1152,
        depth=27,
        mlp_dim=4304,
        num_heads=16,
        projection_dim=2048,
    ),
    paligemma=TransformerConfig(
        width=2048,
        depth=18,
        mlp_dim=16_384,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
    ),
    action_expert=TransformerConfig(
        width=1024,
        depth=18,
        mlp_dim=4096,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
    ),
    vocab_size=257_152,
    action_dim=32,
    action_horizon=50,
    max_token_len=48,
    dtype="bfloat16",
)


SERVER_RUNTIME = RuntimeConfig(
    batch_size=1,
    gradient_checkpointing=True,
    compile_model=True,
    num_workers=2,
)
