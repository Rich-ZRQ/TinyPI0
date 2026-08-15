import pytest
import torch

from configs import TINY_PI0
from pi0.decoder_layer import GemmaDecoderLayer


def make_inputs(
    *,
    batch_size: int,
    sequence_length: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_states = torch.randn(
        batch_size,
        sequence_length,
        width,
    )

    position_ids = torch.arange(
        sequence_length,
    )[None, :].expand(
        batch_size,
        -1,
    )

    attention_mask = torch.ones(
        batch_size,
        sequence_length,
        sequence_length,
        dtype=torch.bool,
    )

    return hidden_states, position_ids, attention_mask


@pytest.mark.parametrize(
    "config",
    [
        TINY_PI0.paligemma,
        TINY_PI0.action_expert,
    ],
)
def test_decoder_layer_output_shapes(config) -> None:
    layer = GemmaDecoderLayer(config)

    hidden_states, position_ids, attention_mask = make_inputs(
        batch_size=2,
        sequence_length=5,
        width=config.width,
    )

    output, probabilities = layer(
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


def test_decoder_layer_matches_explicit_formula() -> None:
    config = TINY_PI0.action_expert
    layer = GemmaDecoderLayer(config)

    hidden_states, position_ids, attention_mask = make_inputs(
        batch_size=2,
        sequence_length=4,
        width=config.width,
    )

    actual, actual_probabilities = layer(
        hidden_states,
        position_ids,
        attention_mask,
    )

    normalized = layer.input_layernorm(hidden_states)

    attention_output, expected_probabilities = layer.self_attn(
        normalized,
        position_ids,
        attention_mask,
    )

    after_attention = hidden_states + attention_output

    normalized = layer.post_attention_layernorm(after_attention)
    mlp_output = layer.mlp(normalized)

    expected = after_attention + mlp_output

    assert torch.allclose(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )

    assert torch.allclose(
        actual_probabilities,
        expected_probabilities,
        atol=1e-6,
        rtol=1e-6,
    )


def test_zero_sublayers_leave_residual_unchanged() -> None:
    config = TINY_PI0.action_expert
    layer = GemmaDecoderLayer(config)

    with torch.no_grad():
        for parameter in layer.self_attn.parameters():
            parameter.zero_()

        for parameter in layer.mlp.parameters():
            parameter.zero_()

    hidden_states, position_ids, attention_mask = make_inputs(
        batch_size=2,
        sequence_length=5,
        width=config.width,
    )

    output, _ = layer(
        hidden_states,
        position_ids,
        attention_mask,
    )

    assert torch.equal(
        output,
        hidden_states,
    )


def test_gradient_flows_through_decoder_layer() -> None:
    config = TINY_PI0.action_expert
    layer = GemmaDecoderLayer(config)

    hidden_states, position_ids, attention_mask = make_inputs(
        batch_size=2,
        sequence_length=5,
        width=config.width,
    )
    hidden_states.requires_grad_(True)

    output, _ = layer(
        hidden_states,
        position_ids,
        attention_mask,
    )

    loss = output.square().mean()
    loss.backward()

    assert hidden_states.grad is not None
    assert layer.self_attn.q_proj.weight.grad is not None
    assert layer.self_attn.k_proj.weight.grad is not None
    assert layer.self_attn.v_proj.weight.grad is not None
    assert layer.self_attn.o_proj.weight.grad is not None

    assert layer.mlp.gate_proj.weight.grad is not None
    assert layer.mlp.up_proj.weight.grad is not None
    assert layer.mlp.down_proj.weight.grad is not None

    assert layer.input_layernorm.weight.grad is not None
    assert layer.post_attention_layernorm.weight.grad is not None

    assert torch.isfinite(hidden_states.grad).all()


def test_wrong_hidden_width_is_rejected() -> None:
    config = TINY_PI0.action_expert
    layer = GemmaDecoderLayer(config)

    hidden_states, position_ids, attention_mask = make_inputs(
        batch_size=2,
        sequence_length=5,
        width=config.width + 1,
    )

    with pytest.raises(
        ValueError,
        match="expected hidden width",
    ):
        layer(
            hidden_states,
            position_ids,
            attention_mask,
        )


def test_non_floating_input_is_rejected() -> None:
    config = TINY_PI0.action_expert
    layer = GemmaDecoderLayer(config)

    hidden_states = torch.ones(
        2,
        5,
        config.width,
        dtype=torch.long,
    )
    position_ids = torch.arange(5)[None, :].expand(2, -1)
    attention_mask = torch.ones(
        2,
        5,
        5,
        dtype=torch.bool,
    )

    with pytest.raises(
        TypeError,
        match="must be floating point",
    ):
        layer(
            hidden_states,
            position_ids,
            attention_mask,
        )
