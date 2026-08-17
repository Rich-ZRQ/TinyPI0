from dataclasses import replace

import pytest

from configs import SO101_TINY, TINY_PI0
from configs.schema import (
    Pi0Config,
    TransformerConfig,
    VisionConfig,
)


def test_external_contract_is_the_same() -> None:
    assert TINY_PI0.action_dim == SO101_TINY.action_dim == 32
    assert TINY_PI0.action_horizon == SO101_TINY.action_horizon == 50
    assert TINY_PI0.max_token_len == SO101_TINY.max_token_len == 48
    assert TINY_PI0.vocab_size == SO101_TINY.vocab_size == 257_152


def test_debug_capacity_is_smaller_than_so101_capacity() -> None:
    assert TINY_PI0.paligemma.width < SO101_TINY.paligemma.width
    assert TINY_PI0.paligemma.depth < SO101_TINY.paligemma.depth
    assert TINY_PI0.action_expert.width < SO101_TINY.action_expert.width


def test_visual_token_count() -> None:
    assert TINY_PI0.vision.num_tokens == 256
    assert SO101_TINY.vision.num_tokens == 256


def test_invalid_expert_depth_is_rejected() -> None:
    wrong_expert = replace(
        TINY_PI0.action_expert,
        depth=3,
    )

    with pytest.raises(ValueError, match="depth"):
        Pi0Config(
            vision=TINY_PI0.vision,
            paligemma=TINY_PI0.paligemma,
            action_expert=wrong_expert,
        )


def test_invalid_patch_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        VisionConfig(
            image_size=225,
            patch_size=14,
            width=128,
            depth=2,
            mlp_dim=256,
            num_heads=4,
            projection_dim=128,
        )


def test_invalid_gqa_grouping_is_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TransformerConfig(
            width=128,
            depth=2,
            mlp_dim=256,
            num_heads=3,
            num_kv_heads=2,
            head_dim=32,
        )


def test_so101_capacity_matches_trained_artifacts() -> None:
    assert SO101_TINY.vision.projection_dim == 1024
    assert SO101_TINY.paligemma.width == 1024
    assert SO101_TINY.paligemma.depth == 8
    assert SO101_TINY.paligemma.mlp_dim == 8192
    assert SO101_TINY.action_expert.width == 512
    assert SO101_TINY.action_expert.depth == 8
    assert SO101_TINY.action_expert.mlp_dim == 2048
