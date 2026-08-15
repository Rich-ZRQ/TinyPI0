import pytest
import torch

from configs import TINY_PI0
from pi0.attention_mask import make_att_2d_masks
from pi0.joint_decoder_layer import JointDecoderLayer


def make_inputs(
    *,
    batch_size: int = 2,
    paligemma_length: int = 5,
    action_length: int = 3,
):
    paligemma_hidden_states = torch.randn(
        batch_size,
        paligemma_length,
        TINY_PI0.paligemma.width,
    )
    action_hidden_states = torch.randn(
        batch_size,
        action_length,
        TINY_PI0.action_expert.width,
    )

    total_length = paligemma_length + action_length

    position_ids = torch.arange(
        total_length,
    )[None, :].expand(
        batch_size,
        -1,
    )

    attention_mask = torch.ones(
        batch_size,
        total_length,
        total_length,
        dtype=torch.bool,
    )

    return (
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    )


def test_experts_use_independent_input_widths() -> None:
    layer = JointDecoderLayer(TINY_PI0)

    paligemma_attention = layer.paligemma_layer.self_attn
    action_attention = layer.action_expert_layer.self_attn

    assert paligemma_attention.q_proj.in_features == 128
    assert action_attention.q_proj.in_features == 64

    assert paligemma_attention.q_proj.out_features == action_attention.q_proj.out_features

    assert paligemma_attention.k_proj.out_features == action_attention.k_proj.out_features


def test_joint_decoder_output_shapes() -> None:
    layer = JointDecoderLayer(TINY_PI0)

    (
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    ) = make_inputs()

    (
        paligemma_output,
        action_output,
        probabilities,
    ) = layer(
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    )

    assert paligemma_output.shape == (paligemma_hidden_states.shape)
    assert action_output.shape == action_hidden_states.shape

    total_length = paligemma_hidden_states.shape[1] + action_hidden_states.shape[1]

    assert probabilities.shape == (
        2,
        TINY_PI0.paligemma.num_heads,
        total_length,
        total_length,
    )


def test_zero_sublayers_preserve_both_residuals() -> None:
    layer = JointDecoderLayer(TINY_PI0)

    modules_to_zero = [
        layer.paligemma_layer.self_attn,
        layer.paligemma_layer.mlp,
        layer.action_expert_layer.self_attn,
        layer.action_expert_layer.mlp,
    ]

    with torch.no_grad():
        for module in modules_to_zero:
            for parameter in module.parameters():
                parameter.zero_()

    (
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    ) = make_inputs()

    paligemma_output, action_output, _ = layer(
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    )

    assert torch.equal(
        paligemma_output,
        paligemma_hidden_states,
    )
    assert torch.equal(
        action_output,
        action_hidden_states,
    )


def test_action_tokens_influence_paligemma_tokens() -> None:
    torch.manual_seed(0)

    layer = JointDecoderLayer(TINY_PI0)

    (
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    ) = make_inputs(
        batch_size=1,
    )

    changed_action_hidden_states = action_hidden_states + 10.0

    first_output, _, _ = layer(
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    )

    second_output, _, _ = layer(
        paligemma_hidden_states,
        changed_action_hidden_states,
        position_ids,
        attention_mask,
    )

    assert not torch.allclose(
        first_output,
        second_output,
    )


def test_mask_can_block_action_to_paligemma_flow() -> None:
    torch.manual_seed(0)

    layer = JointDecoderLayer(TINY_PI0)

    (
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    ) = make_inputs(
        batch_size=1,
        paligemma_length=5,
        action_length=3,
    )

    attention_mask[:, :5, 5:] = False

    changed_action_hidden_states = action_hidden_states + 10.0

    first_output, _, _ = layer(
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    )

    second_output, _, _ = layer(
        paligemma_hidden_states,
        changed_action_hidden_states,
        position_ids,
        attention_mask,
    )

    assert torch.allclose(
        first_output,
        second_output,
        atol=1e-6,
        rtol=1e-6,
    )


def test_cross_expert_gradient_flow() -> None:
    layer = JointDecoderLayer(TINY_PI0)

    (
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    ) = make_inputs()

    paligemma_hidden_states.requires_grad_(True)
    action_hidden_states.requires_grad_(True)

    paligemma_output, _, _ = layer(
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    )

    loss = paligemma_output.square().mean()
    loss.backward()

    assert paligemma_hidden_states.grad is not None
    assert action_hidden_states.grad is not None

    assert action_hidden_states.grad.abs().sum().item() > 0

    assert layer.action_expert_layer.self_attn.k_proj.weight.grad is not None
    assert layer.action_expert_layer.self_attn.v_proj.weight.grad is not None


def test_mismatched_batch_sizes_are_rejected() -> None:
    layer = JointDecoderLayer(TINY_PI0)

    (
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    ) = make_inputs()

    wrong_action_hidden_states = action_hidden_states[:1]

    with pytest.raises(
        ValueError,
        match="same batch size",
    ):
        layer(
            paligemma_hidden_states,
            wrong_action_hidden_states,
            position_ids,
            attention_mask,
        )


def test_wrong_joint_mask_shape_is_rejected() -> None:
    layer = JointDecoderLayer(TINY_PI0)

    (
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    ) = make_inputs()

    wrong_attention_mask = attention_mask[:, :-1, :-1]

    with pytest.raises(
        ValueError,
        match="attention_mask must have shape",
    ):
        layer(
            paligemma_hidden_states,
            action_hidden_states,
            position_ids,
            wrong_attention_mask,
        )


def test_cached_suffix_matches_joint_forward() -> None:
    torch.manual_seed(0)

    layer = JointDecoderLayer(TINY_PI0)
    layer.eval()

    batch_size = 2
    prefix_length = 6
    suffix_length = 4
    total_length = prefix_length + suffix_length

    prefix = torch.randn(
        batch_size,
        prefix_length,
        TINY_PI0.paligemma.width,
    )
    suffix = torch.randn(
        batch_size,
        suffix_length,
        TINY_PI0.action_expert.width,
    )

    pad_masks = torch.ones(
        batch_size,
        total_length,
        dtype=torch.bool,
    )

    block_masks = torch.zeros(
        batch_size,
        total_length,
        dtype=torch.bool,
    )

    # suffix[0]是state token，开始一个新块。
    block_masks[:, prefix_length] = True

    # suffix[1:]是action tokens，再开始一个新块。
    block_masks[:, prefix_length + 1] = True

    attention_mask = make_att_2d_masks(
        pad_masks,
        block_masks,
    )

    position_ids = (
        torch.cumsum(
            pad_masks.to(torch.int64),
            dim=1,
        )
        - 1
    )

    # 原始的一次性联合前向。
    joint_prefix, joint_suffix, _ = layer(
        prefix,
        suffix,
        position_ids,
        attention_mask,
    )

    # 第一步：单独计算prefix并生成缓存。
    (
        cached_prefix,
        prefix_key,
        prefix_value,
    ) = layer.encode_prefix(
        prefix_hidden_states=prefix,
        position_ids=position_ids[
            :,
            :prefix_length,
        ],
        attention_mask=attention_mask[
            :,
            :prefix_length,
            :prefix_length,
        ],
    )

    # 第二步：suffix只读取缓存，不重新计算prefix。
    cached_suffix = layer.decode_suffix(
        suffix_hidden_states=suffix,
        prefix_key=prefix_key,
        prefix_value=prefix_value,
        position_ids=position_ids[
            :,
            prefix_length:,
        ],
        attention_mask=attention_mask[
            :,
            prefix_length:,
            :,
        ],
    )

    torch.testing.assert_close(
        cached_prefix,
        joint_prefix,
        rtol=1e-5,
        atol=1e-5,
    )

    torch.testing.assert_close(
        cached_suffix,
        joint_suffix,
        rtol=1e-5,
        atol=1e-5,
    )

    assert prefix_key.shape == (
        batch_size,
        TINY_PI0.paligemma.num_kv_heads,
        prefix_length,
        TINY_PI0.paligemma.head_dim,
    )

    assert prefix_value.shape == prefix_key.shape
