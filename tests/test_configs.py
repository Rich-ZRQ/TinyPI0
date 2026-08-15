from dataclasses import replace

import pytest

from configs import FULL_PI0, SO101_LARGE, SO101_RECOMMENDED, TINY_PI0
from configs.schema import (
    Pi0Config,
    TransformerConfig,
    VisionConfig,
)


def test_external_contract_is_the_same() -> None:
    assert TINY_PI0.action_dim == FULL_PI0.action_dim == 32
    assert TINY_PI0.action_horizon == FULL_PI0.action_horizon == 50
    assert TINY_PI0.max_token_len == FULL_PI0.max_token_len == 48
    assert TINY_PI0.vocab_size == FULL_PI0.vocab_size == 257_152


def test_tiny_capacity_is_smaller() -> None:
    assert TINY_PI0.vision.width < FULL_PI0.vision.width
    assert TINY_PI0.vision.depth < FULL_PI0.vision.depth
    assert TINY_PI0.paligemma.width < FULL_PI0.paligemma.width
    assert TINY_PI0.paligemma.depth < FULL_PI0.paligemma.depth
    assert TINY_PI0.action_expert.width < FULL_PI0.action_expert.width


def test_visual_token_count() -> None:
    assert TINY_PI0.vision.num_tokens == 256
    assert FULL_PI0.vision.num_tokens == 256


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


def test_so101_profiles_keep_the_pi0_external_contract() -> None:
    for config in (SO101_RECOMMENDED, SO101_LARGE):
        assert config.action_dim == 32
        assert config.action_horizon == 50
        assert config.max_token_len == 48
        assert config.vision.num_tokens == 256


def test_so101_large_has_more_capacity() -> None:
    assert SO101_LARGE.paligemma.width > SO101_RECOMMENDED.paligemma.width
    assert SO101_LARGE.action_expert.width > SO101_RECOMMENDED.action_expert.width
    assert SO101_LARGE.paligemma.mlp_dim > SO101_RECOMMENDED.paligemma.mlp_dim
