import pytest
import torch

from configs import TINY_PI0
from pi0.core import Pi0Core
from pi0.flow_matching import (
    flow_matching_loss,
    make_flow_matching_target,
)


def make_inputs(
    *,
    batch_size: int = 2,
    prefix_length: int = 6,
):
    prefix_tokens = torch.randn(
        batch_size,
        prefix_length,
        TINY_PI0.paligemma.width,
    )

    prefix_pad_masks = torch.ones(
        batch_size,
        prefix_length,
        dtype=torch.bool,
    )

    prefix_att_masks = torch.zeros(
        batch_size,
        prefix_length,
        dtype=torch.bool,
    )

    state = torch.randn(
        batch_size,
        TINY_PI0.action_dim,
    )

    actions = torch.randn(
        batch_size,
        TINY_PI0.action_horizon,
        TINY_PI0.action_dim,
    )

    noise = torch.randn_like(actions)
    timestep = torch.rand(batch_size)

    return (
        prefix_tokens,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        actions,
        noise,
        timestep,
    )


def test_predict_velocity_shape() -> None:
    model = Pi0Core(TINY_PI0)

    (
        prefix_tokens,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        actions,
        _,
        timestep,
    ) = make_inputs()

    velocity = model.predict_velocity(
        prefix_tokens,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        actions,
        timestep,
    )

    assert velocity.shape == actions.shape


def test_training_loss_shape() -> None:
    model = Pi0Core(TINY_PI0)

    (
        prefix_tokens,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        actions,
        noise,
        timestep,
    ) = make_inputs()

    loss = model.training_loss(
        prefix_tokens,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        actions,
        noise=noise,
        timestep=timestep,
    )

    assert loss.shape == actions.shape
    assert torch.isfinite(loss).all()


def test_training_loss_matches_explicit_flow_path() -> None:
    model = Pi0Core(TINY_PI0)

    (
        prefix_tokens,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        actions,
        noise,
        timestep,
    ) = make_inputs()

    actual_loss = model.training_loss(
        prefix_tokens,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        actions,
        noise=noise,
        timestep=timestep,
    )

    noisy_actions, target_velocity = make_flow_matching_target(
        actions,
        noise,
        timestep,
    )

    predicted_velocity = model.predict_velocity(
        prefix_tokens,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        noisy_actions,
        timestep,
    )

    expected_loss = flow_matching_loss(
        predicted_velocity,
        target_velocity,
    )

    assert torch.allclose(
        actual_loss,
        expected_loss,
        atol=1e-6,
        rtol=1e-6,
    )


def test_action_loss_reaches_prefix_and_state_projection() -> None:
    model = Pi0Core(TINY_PI0)

    (
        prefix_tokens,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        actions,
        noise,
        timestep,
    ) = make_inputs()

    prefix_tokens.requires_grad_(True)

    loss = model.training_loss(
        prefix_tokens,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        actions,
        noise=noise,
        timestep=timestep,
    ).mean()

    loss.backward()

    assert prefix_tokens.grad is not None
    assert prefix_tokens.grad.abs().sum().item() > 0

    state_gradient = model.action_embedding.state_proj.weight.grad
    assert state_gradient is not None
    assert state_gradient.abs().sum().item() > 0


def test_loss_reaches_both_experts() -> None:
    model = Pi0Core(TINY_PI0)

    (
        prefix_tokens,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        actions,
        noise,
        timestep,
    ) = make_inputs()

    loss = model.training_loss(
        prefix_tokens,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        actions,
        noise=noise,
        timestep=timestep,
    ).mean()

    loss.backward()

    first_layer = model.transformer.layers[0]

    assert first_layer.paligemma_layer.self_attn.k_proj.weight.grad is not None
    assert first_layer.paligemma_layer.self_attn.v_proj.weight.grad is not None
    assert first_layer.action_expert_layer.self_attn.q_proj.weight.grad is not None
    assert model.action_embedding.action_out_proj.weight.grad is not None


def test_padded_prefix_token_cannot_affect_velocity() -> None:
    torch.manual_seed(0)

    model = Pi0Core(TINY_PI0)

    (
        prefix_tokens,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        actions,
        _,
        timestep,
    ) = make_inputs(
        batch_size=1,
        prefix_length=6,
    )

    prefix_pad_masks[:, -1] = False

    changed_prefix_tokens = prefix_tokens.clone()
    changed_prefix_tokens[:, -1] += 1000.0

    first_velocity = model.predict_velocity(
        prefix_tokens,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        actions,
        timestep,
    )

    second_velocity = model.predict_velocity(
        changed_prefix_tokens,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        actions,
        timestep,
    )

    assert torch.allclose(
        first_velocity,
        second_velocity,
        atol=1e-5,
        rtol=1e-5,
    )


def test_wrong_prefix_width_is_rejected() -> None:
    model = Pi0Core(TINY_PI0)

    (
        prefix_tokens,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        actions,
        _,
        timestep,
    ) = make_inputs()

    wrong_prefix_tokens = torch.randn(
        prefix_tokens.shape[0],
        prefix_tokens.shape[1],
        TINY_PI0.paligemma.width + 1,
    )

    with pytest.raises(
        ValueError,
        match="prefix token width",
    ):
        model.predict_velocity(
            wrong_prefix_tokens,
            prefix_pad_masks,
            prefix_att_masks,
            state,
            actions,
            timestep,
        )


def test_wrong_prefix_mask_dtype_is_rejected() -> None:
    model = Pi0Core(TINY_PI0)

    (
        prefix_tokens,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        actions,
        _,
        timestep,
    ) = make_inputs()

    wrong_pad_masks = prefix_pad_masks.to(torch.float32)

    with pytest.raises(
        TypeError,
        match="prefix_pad_masks must be bool",
    ):
        model.predict_velocity(
            prefix_tokens,
            wrong_pad_masks,
            prefix_att_masks,
            state,
            actions,
            timestep,
        )


def test_cached_velocity_matches_joint_velocity() -> None:
    torch.manual_seed(0)

    model = Pi0Core(TINY_PI0)
    model.eval()

    (
        prefix_tokens,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        actions,
        _,
        timestep,
    ) = make_inputs()

    prefix_pad_masks[:, -2:] = False

    joint_velocity = model.predict_velocity(
        prefix_tokens,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        actions,
        timestep,
    )

    prefix_cache = model.prefill_prefix(
        prefix_tokens,
        prefix_pad_masks,
        prefix_att_masks,
        state,
    )
    cached_velocity = model.predict_velocity_with_cache(
        prefix_cache,
        prefix_pad_masks,
        state,
        actions,
        timestep,
    )

    torch.testing.assert_close(
        cached_velocity,
        joint_velocity,
        rtol=1e-5,
        atol=1e-5,
    )
