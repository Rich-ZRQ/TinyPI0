import pytest
import torch

from configs import TINY_PI0
from pi0.attention import GemmaAttention, repeat_kv
from pi0.attention_mask import make_att_2d_masks


def test_repeat_kv_heads() -> None:
    hidden_states = torch.tensor(
        [
            [
                [[1.0, 2.0]],
                [[3.0, 4.0]],
            ]
        ]
    )

    repeated = repeat_kv(
        hidden_states,
        num_repeats=3,
    )

    assert repeated.shape == (1, 6, 1, 2)

    assert torch.equal(
        repeated[:, 0],
        hidden_states[:, 0],
    )
    assert torch.equal(
        repeated[:, 1],
        hidden_states[:, 0],
    )
    assert torch.equal(
        repeated[:, 2],
        hidden_states[:, 0],
    )

    assert torch.equal(
        repeated[:, 3],
        hidden_states[:, 1],
    )


def test_action_expert_projection_shapes() -> None:
    config = TINY_PI0.action_expert
    attention = GemmaAttention(config)

    hidden_states = torch.randn(
        2,
        5,
        config.width,
    )

    query, key, value = attention.project_qkv(hidden_states)

    assert query.shape == (
        2,
        config.num_heads,
        5,
        config.head_dim,
    )
    assert key.shape == (
        2,
        config.num_kv_heads,
        5,
        config.head_dim,
    )
    assert value.shape == key.shape


def test_attention_output_shapes() -> None:
    config = TINY_PI0.action_expert
    attention = GemmaAttention(config)

    hidden_states = torch.randn(
        2,
        5,
        config.width,
    )
    position_ids = torch.arange(5).repeat(2, 1)
    attention_mask = torch.ones(
        2,
        5,
        5,
        dtype=torch.bool,
    )

    output, probabilities = attention(
        hidden_states,
        position_ids,
        attention_mask,
    )

    assert output.shape == hidden_states.shape
    assert probabilities.shape == (
        2,
        config.num_heads,
        5,
        5,
    )


def test_causal_mask_blocks_future_tokens() -> None:
    config = TINY_PI0.action_expert
    attention = GemmaAttention(config)

    hidden_states = torch.randn(
        1,
        4,
        config.width,
    )
    position_ids = torch.arange(4)[None, :]

    pad_masks = torch.ones(
        1,
        4,
        dtype=torch.bool,
    )
    att_masks = torch.ones(
        1,
        4,
        dtype=torch.bool,
    )
    attention_mask = make_att_2d_masks(
        pad_masks,
        att_masks,
    )

    _, probabilities = attention(
        hidden_states,
        position_ids,
        attention_mask,
    )

    assert torch.equal(
        probabilities[:, :, 0, 1:],
        torch.zeros_like(probabilities[:, :, 0, 1:]),
    )

    assert torch.equal(
        probabilities[:, :, 1, 2:],
        torch.zeros_like(probabilities[:, :, 1, 2:]),
    )

    row_sums = probabilities.sum(dim=-1)

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-6,
        rtol=1e-6,
    )


def test_gradient_flows_through_attention() -> None:
    config = TINY_PI0.action_expert
    attention = GemmaAttention(config)

    hidden_states = torch.randn(
        2,
        5,
        config.width,
        requires_grad=True,
    )
    position_ids = torch.arange(5).repeat(2, 1)
    attention_mask = torch.ones(
        2,
        5,
        5,
        dtype=torch.bool,
    )

    output, _ = attention(
        hidden_states,
        position_ids,
        attention_mask,
    )

    loss = output.square().mean()
    loss.backward()

    assert hidden_states.grad is not None
    assert attention.q_proj.weight.grad is not None
    assert attention.k_proj.weight.grad is not None
    assert attention.v_proj.weight.grad is not None
    assert attention.o_proj.weight.grad is not None


def test_action_expert_width_can_differ_from_q_width() -> None:
    config = TINY_PI0.action_expert
    attention = GemmaAttention(config)

    assert config.width == 64
    assert config.num_heads * config.head_dim == 128

    hidden_states = torch.randn(
        1,
        3,
        config.width,
    )
    position_ids = torch.arange(3)[None, :]
    attention_mask = torch.ones(
        1,
        3,
        3,
        dtype=torch.bool,
    )

    output, _ = attention(
        hidden_states,
        position_ids,
        attention_mask,
    )

    assert output.shape == (1, 3, 64)


def test_wrong_hidden_width_is_rejected() -> None:
    config = TINY_PI0.action_expert
    attention = GemmaAttention(config)

    hidden_states = torch.randn(
        2,
        5,
        config.width + 1,
    )

    with pytest.raises(
        ValueError,
        match="expected hidden width",
    ):
        attention.project_qkv(hidden_states)
