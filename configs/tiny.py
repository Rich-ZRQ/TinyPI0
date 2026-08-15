"""Minimal pi0 profile for unit tests and fast architecture debugging.

Use ``SO101_RECOMMENDED`` for real two-camera SO-ARM101 experiments. Keeping
this profile small makes the complete test suite practical on a laptop.
"""

from configs.schema import (
    Pi0Config,
    RuntimeConfig,
    TransformerConfig,
    VisionConfig,
)

TINY_PI0 = Pi0Config(
    vision=VisionConfig(
        image_size=224,
        patch_size=14,
        width=128,
        depth=2,
        mlp_dim=256,
        num_heads=4,
        projection_dim=128,
    ),
    paligemma=TransformerConfig(
        width=128,
        depth=2,
        mlp_dim=256,
        num_heads=4,
        num_kv_heads=1,
        head_dim=32,
    ),
    action_expert=TransformerConfig(
        width=64,
        depth=2,
        mlp_dim=128,
        num_heads=4,
        num_kv_heads=1,
        head_dim=32,
    ),
    vocab_size=257_152,
    action_dim=32,
    action_horizon=50,
    max_token_len=48,
    dtype="float32",
)


LOCAL_RUNTIME = RuntimeConfig(
    batch_size=1,
    gradient_checkpointing=False,
    compile_model=False,
    num_workers=0,
)
