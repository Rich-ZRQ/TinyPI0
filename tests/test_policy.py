from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from configs.tiny import TINY_PI0
from pi0.normalization import NormStats, Pi0Normalizer
from pi0.policy import Pi0Policy
from pi0.types import IMAGE_KEYS, Observation


class FakePrefixEncoder(nn.Module):
    """Small replacement for the frozen PaliGemma frontend."""

    def __init__(self, source_width: int = 16) -> None:
        super().__init__()

        self.config = SimpleNamespace(
            text_config=SimpleNamespace(
                hidden_size=source_width,
            )
        )

        # 给 Pi0Policy 提供 device 和 dtype 参考。
        self.anchor = nn.Parameter(
            torch.zeros(1),
            requires_grad=False,
        )

        self.source_width = source_width
        self.image_token_count = 4

    def encode_images(
        self,
        pixel_values: Tensor,
    ) -> Tensor:
        batch_size = pixel_values.shape[0]

        image_value = pixel_values.mean(
            dim=(1, 2, 3),
            keepdim=False,
        )

        return image_value[:, None, None].expand(
            batch_size,
            self.image_token_count,
            self.source_width,
        )

    def embed_text(
        self,
        input_ids: Tensor,
    ) -> Tensor:
        token_values = input_ids.to(torch.float32)

        return token_values[:, :, None].expand(
            input_ids.shape[0],
            input_ids.shape[1],
            self.source_width,
        )


def make_observation(
    batch_size: int = 2,
) -> Observation:
    images = {
        key: torch.ones(
            batch_size,
            3,
            224,
            224,
        )
        for key in IMAGE_KEYS
    }

    # 模拟只有两个有效相机。
    image_masks = {
        IMAGE_KEYS[0]: torch.ones(
            batch_size,
            dtype=torch.bool,
        ),
        IMAGE_KEYS[1]: torch.ones(
            batch_size,
            dtype=torch.bool,
        ),
        IMAGE_KEYS[2]: torch.zeros(
            batch_size,
            dtype=torch.bool,
        ),
    }

    tokenized_prompt = torch.ones(
        batch_size,
        TINY_PI0.max_token_len,
        dtype=torch.long,
    )

    tokenized_prompt_mask = torch.zeros(
        batch_size,
        TINY_PI0.max_token_len,
        dtype=torch.bool,
    )
    tokenized_prompt_mask[:, :5] = True

    return Observation(
        images=images,
        image_masks=image_masks,
        state=torch.zeros(
            batch_size,
            TINY_PI0.action_dim,
        ),
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )


def test_prefix_assembly_skips_missing_camera() -> None:
    encoder = FakePrefixEncoder()
    policy = Pi0Policy(TINY_PI0, encoder)

    observation = make_observation(batch_size=2)

    tokens, pad_masks, att_masks = policy.prefix_embedding(observation)

    # 两个相机各4个token，加上固定48个文本位置。
    expected_length = 2 * encoder.image_token_count + TINY_PI0.max_token_len

    assert tokens.shape == (
        2,
        expected_length,
        TINY_PI0.paligemma.width,
    )
    assert pad_masks.shape == (2, expected_length)
    assert att_masks.shape == (2, expected_length)

    # 8个视觉token + 5个有效文本token。
    assert torch.all(pad_masks.sum(dim=1) == 13)
    assert not torch.any(att_masks)


def test_policy_compute_loss_and_backward() -> None:
    encoder = FakePrefixEncoder()
    policy = Pi0Policy(TINY_PI0, encoder)

    observation = make_observation(batch_size=2)

    actions = torch.randn(
        2,
        TINY_PI0.action_horizon,
        TINY_PI0.action_dim,
    )

    loss = policy.compute_loss(
        observation,
        actions,
    )

    assert loss.shape == (
        2,
        TINY_PI0.action_horizon,
    )
    assert torch.isfinite(loss).all()

    loss.mean().backward()

    projection_gradient = policy.prefix_embedding.input_projection.weight.grad

    assert projection_gradient is not None
    assert torch.isfinite(projection_gradient).all()
    assert encoder.anchor.grad is None


def test_policy_accepts_fixed_flow_targets_for_validation() -> None:
    policy = Pi0Policy(TINY_PI0, FakePrefixEncoder())
    observation = make_observation(batch_size=1)
    actions = torch.randn(1, TINY_PI0.action_horizon, TINY_PI0.action_dim)
    noise = torch.zeros_like(actions)
    timestep = torch.full((1,), 0.5)

    first = policy.compute_loss(
        observation,
        actions,
        noise=noise,
        timestep=timestep,
    )
    second = policy.compute_loss(
        observation,
        actions,
        noise=noise,
        timestep=timestep,
    )

    assert torch.equal(first, second)


def test_trainable_dtype_comes_from_model_config() -> None:
    encoder = FakePrefixEncoder().to(dtype=torch.bfloat16)
    policy = Pi0Policy(TINY_PI0, encoder)

    assert policy.model_dtype == torch.float32
    assert policy.prefix_embedding.input_projection.weight.dtype == torch.float32
    assert next(policy.prefix_embedding.prefix_encoder.parameters()).dtype == torch.bfloat16


def test_training_can_keep_master_parameters_float32() -> None:
    encoder = FakePrefixEncoder().to(dtype=torch.bfloat16)
    policy = Pi0Policy(
        TINY_PI0,
        encoder,
        trainable_dtype=torch.float32,
    )

    assert all(parameter.dtype == torch.float32 for parameter in policy.parameters() if parameter.requires_grad)
    assert next(policy.prefix_embedding.prefix_encoder.parameters()).dtype == torch.bfloat16


def test_action_dimension_mask_excludes_robot_padding() -> None:
    policy = Pi0Policy(TINY_PI0, FakePrefixEncoder())
    observation = make_observation(batch_size=2)
    actions = torch.zeros(2, TINY_PI0.action_horizon, TINY_PI0.action_dim)
    dimension_mask = torch.zeros(2, TINY_PI0.action_dim, dtype=torch.bool)
    dimension_mask[:, :6] = True

    captured_noise = None

    def fixed_element_loss(**kwargs: Tensor) -> Tensor:
        nonlocal captured_noise
        captured_noise = kwargs["noise"]
        loss = torch.full_like(kwargs["actions"], 100.0)
        loss[..., :6] = 2.0
        return loss

    policy.core.training_loss = fixed_element_loss
    loss = policy.compute_loss(
        observation,
        actions,
        action_dim_mask=dimension_mask,
    )

    assert torch.equal(loss, torch.full_like(loss, 2.0))
    assert captured_noise is not None
    assert torch.count_nonzero(captured_noise[..., 6:]) == 0


def test_sampling_keeps_padding_dimensions_zero() -> None:
    state_stats = NormStats(
        mean=torch.zeros(6),
        std=torch.ones(6),
    )
    action_stats = NormStats(
        mean=torch.zeros(6),
        std=torch.ones(6),
    )
    policy = Pi0Policy(
        TINY_PI0,
        FakePrefixEncoder(),
        normalizer=Pi0Normalizer(state_stats, action_stats),
    )
    observation = make_observation(batch_size=1)

    def predict_unit_velocity(**kwargs: Tensor) -> Tensor:
        return torch.ones_like(kwargs["noisy_actions"])

    policy.core.predict_velocity_with_cache = predict_unit_velocity
    actions = policy.sample_actions(
        observation,
        num_steps=2,
        noise=torch.ones(1, TINY_PI0.action_horizon, TINY_PI0.action_dim),
    )

    assert torch.count_nonzero(actions[..., 6:]) == 0


def test_trainable_linears_use_gemma_initialization() -> None:
    torch.manual_seed(0)
    policy = Pi0Policy(TINY_PI0, FakePrefixEncoder())
    weights = policy.core.transformer.layers[0].paligemma_layer.self_attn.q_proj.weight

    assert weights.mean().item() == pytest.approx(0.0, abs=1e-3)
    assert weights.std().item() == pytest.approx(0.02, abs=1e-3)
    assert torch.count_nonzero(policy.core.action_embedding.action_out_proj.weight) == 0


def test_policy_sample_actions() -> None:
    encoder = FakePrefixEncoder()
    policy = Pi0Policy(TINY_PI0, encoder)
    policy.eval()

    observation = make_observation(batch_size=2)

    initial_noise = torch.randn(
        2,
        TINY_PI0.action_horizon,
        TINY_PI0.action_dim,
    )

    actions = policy.sample_actions(
        observation,
        num_steps=2,
        noise=initial_noise,
    )

    assert actions.shape == initial_noise.shape
    assert actions.dtype == torch.float32
    assert torch.isfinite(actions).all()


def test_invalid_sampling_steps_are_rejected() -> None:
    encoder = FakePrefixEncoder()
    policy = Pi0Policy(TINY_PI0, encoder)

    observation = make_observation(batch_size=1)

    try:
        policy.sample_actions(
            observation,
            num_steps=0,
        )
    except ValueError as error:
        assert "num_steps must be positive" in str(error)
    else:
        raise AssertionError("Expected num_steps=0 to raise ValueError")


def test_policy_unnormalizes_sampled_actions() -> None:
    encoder = FakePrefixEncoder()

    state_stats = NormStats(
        mean=torch.full(
            (TINY_PI0.action_dim,),
            10.0,
        ),
        std=torch.full(
            (TINY_PI0.action_dim,),
            2.0,
        ),
    )

    action_stats = NormStats(
        mean=torch.full(
            (TINY_PI0.action_dim,),
            5.0,
        ),
        std=torch.full(
            (TINY_PI0.action_dim,),
            3.0,
        ),
    )

    normalizer = Pi0Normalizer(
        state_stats=state_stats,
        action_stats=action_stats,
    )

    policy = Pi0Policy(
        TINY_PI0,
        encoder,
        normalizer=normalizer,
    )
    policy.eval()

    observation = make_observation(batch_size=1)

    def predict_zero_velocity(**kwargs: Tensor) -> Tensor:
        return torch.zeros_like(kwargs["noisy_actions"])

    policy.core.predict_velocity_with_cache = predict_zero_velocity

    normalized_noise = torch.zeros(
        1,
        TINY_PI0.action_horizon,
        TINY_PI0.action_dim,
    )

    actions = policy.sample_actions(
        observation,
        num_steps=2,
        noise=normalized_noise,
    )

    # 归一化空间中的0，对应真实空间中的action mean。
    assert torch.allclose(
        actions,
        torch.full_like(actions, 5.0),
        atol=1e-5,
    )
