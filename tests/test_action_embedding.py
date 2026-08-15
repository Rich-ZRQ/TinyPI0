import pytest
import torch
from torch.nn import functional as F

from configs import TINY_PI0
from pi0.action_embedding import Pi0ActionEmbedding
from pi0.time_embedding import (
    create_sinusoidal_pos_embedding,
)


def make_inputs(
    *,
    batch_size: int = 2,
):
    state = torch.randn(
        batch_size,
        TINY_PI0.action_dim,
    )

    noisy_actions = torch.randn(
        batch_size,
        TINY_PI0.action_horizon,
        TINY_PI0.action_dim,
    )

    timestep = torch.rand(batch_size)

    return state, noisy_actions, timestep


def test_projection_shapes_match_official_contract() -> None:
    embedding = Pi0ActionEmbedding(TINY_PI0)

    width = TINY_PI0.action_expert.width
    action_dim = TINY_PI0.action_dim

    assert embedding.state_proj.weight.shape == (
        width,
        action_dim,
    )
    assert embedding.action_in_proj.weight.shape == (
        width,
        action_dim,
    )
    assert embedding.action_time_mlp_in.weight.shape == (
        width,
        2 * width,
    )
    assert embedding.action_time_mlp_out.weight.shape == (
        width,
        width,
    )
    assert embedding.action_out_proj.weight.shape == (
        action_dim,
        width,
    )

    assert embedding.state_proj.bias is not None
    assert embedding.action_in_proj.bias is not None
    assert embedding.action_time_mlp_in.bias is not None
    assert embedding.action_time_mlp_out.bias is not None
    assert embedding.action_out_proj.bias is not None


def test_suffix_shapes_and_masks() -> None:
    embedding = Pi0ActionEmbedding(TINY_PI0)

    state, noisy_actions, timestep = make_inputs()

    suffix_tokens, pad_masks, att_masks = embedding(
        state,
        noisy_actions,
        timestep,
    )

    suffix_length = 1 + TINY_PI0.action_horizon

    assert suffix_tokens.shape == (
        2,
        suffix_length,
        TINY_PI0.action_expert.width,
    )
    assert pad_masks.shape == (2, suffix_length)
    assert att_masks.shape == (2, suffix_length)

    assert pad_masks.dtype == torch.bool
    assert att_masks.dtype == torch.bool

    assert pad_masks.all()
    assert att_masks[:, 0].all()
    assert att_masks[:, 1].all()
    assert not att_masks[:, 2:].any()


def test_embedding_matches_explicit_formula() -> None:
    embedding = Pi0ActionEmbedding(TINY_PI0)

    state, noisy_actions, timestep = make_inputs()

    actual, _, _ = embedding(
        state,
        noisy_actions,
        timestep,
    )

    state_token = embedding.state_proj(state)[:, None, :]

    action_embedding = embedding.action_in_proj(noisy_actions)

    time_embedding = create_sinusoidal_pos_embedding(
        timestep,
        TINY_PI0.action_expert.width,
        min_period=4e-3,
        max_period=4.0,
    )

    time_embedding = time_embedding[:, None, :].expand_as(action_embedding)

    fused = torch.cat(
        [action_embedding, time_embedding],
        dim=-1,
    )
    action_tokens = embedding.action_time_mlp_out(F.silu(embedding.action_time_mlp_in(fused)))

    expected = torch.cat(
        [state_token, action_tokens],
        dim=1,
    )

    assert torch.allclose(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_timestep_changes_actions_but_not_state_token() -> None:
    torch.manual_seed(0)

    embedding = Pi0ActionEmbedding(TINY_PI0)

    state, noisy_actions, timestep = make_inputs()

    first, _, _ = embedding(
        state,
        noisy_actions,
        timestep,
    )
    second, _, _ = embedding(
        state,
        noisy_actions,
        1.0 - timestep,
    )

    assert torch.equal(
        first[:, :1],
        second[:, :1],
    )
    assert not torch.allclose(
        first[:, 1:],
        second[:, 1:],
    )


def test_project_velocity_shape() -> None:
    embedding = Pi0ActionEmbedding(TINY_PI0)

    action_hidden_states = torch.randn(
        2,
        TINY_PI0.action_horizon,
        TINY_PI0.action_expert.width,
    )

    velocity = embedding.project_velocity(action_hidden_states)

    assert velocity.shape == (
        2,
        TINY_PI0.action_horizon,
        TINY_PI0.action_dim,
    )


def test_gradients_reach_all_projections() -> None:
    embedding = Pi0ActionEmbedding(TINY_PI0)

    state, noisy_actions, timestep = make_inputs()

    suffix_tokens, _, _ = embedding(
        state,
        noisy_actions,
        timestep,
    )

    velocity = embedding.project_velocity(suffix_tokens[:, 1:])

    velocity.square().mean().backward()

    assert embedding.state_proj.weight.grad is not None
    assert torch.count_nonzero(embedding.state_proj.weight.grad).item() == 0
    assert embedding.action_in_proj.weight.grad is not None
    assert embedding.action_time_mlp_in.weight.grad is not None
    assert embedding.action_time_mlp_out.weight.grad is not None
    assert embedding.action_out_proj.weight.grad is not None


@pytest.mark.parametrize(
    ("input_name", "replacement"),
    [
        (
            "state",
            torch.randn(2, TINY_PI0.action_dim + 1),
        ),
        (
            "noisy_actions",
            torch.randn(
                2,
                TINY_PI0.action_horizon - 1,
                TINY_PI0.action_dim,
            ),
        ),
        (
            "timestep",
            torch.randn(2, 1),
        ),
    ],
)
def test_invalid_input_shapes_are_rejected(
    input_name,
    replacement,
) -> None:
    embedding = Pi0ActionEmbedding(TINY_PI0)

    state, noisy_actions, timestep = make_inputs()

    inputs = {
        "state": state,
        "noisy_actions": noisy_actions,
        "timestep": timestep,
    }
    inputs[input_name] = replacement

    with pytest.raises(ValueError):
        embedding(
            inputs["state"],
            inputs["noisy_actions"],
            inputs["timestep"],
        )
