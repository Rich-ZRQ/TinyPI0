import torch

from configs import TINY_PI0
from pi0.attention_mask import make_att_2d_masks
from pi0.joint_transformer import JointTransformer


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


def test_transformer_uses_configured_depth() -> None:
    model = JointTransformer(TINY_PI0)

    assert len(model.layers) == TINY_PI0.paligemma.depth
    assert len(model.layers) == TINY_PI0.action_expert.depth
    assert len(model.layers) == 2


def test_each_layer_has_two_independent_experts() -> None:
    model = JointTransformer(TINY_PI0)

    for layer in model.layers:
        paligemma_attention = layer.paligemma_layer.self_attn
        action_attention = layer.action_expert_layer.self_attn

        assert paligemma_attention.q_proj.in_features == (TINY_PI0.paligemma.width)
        assert action_attention.q_proj.in_features == (TINY_PI0.action_expert.width)

        assert paligemma_attention.q_proj.weight.data_ptr() != action_attention.q_proj.weight.data_ptr()


def test_transformer_output_shapes() -> None:
    model = JointTransformer(TINY_PI0)

    (
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    ) = make_inputs()

    paligemma_output, action_output = model(
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    )

    assert paligemma_output.shape == (paligemma_hidden_states.shape)
    assert action_output.shape == action_hidden_states.shape


def test_forward_matches_explicit_layer_stack() -> None:
    model = JointTransformer(TINY_PI0)

    (
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    ) = make_inputs()

    actual_paligemma, actual_action = model(
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    )

    expected_paligemma = paligemma_hidden_states
    expected_action = action_hidden_states

    for layer in model.layers:
        (
            expected_paligemma,
            expected_action,
            _,
        ) = layer(
            expected_paligemma,
            expected_action,
            position_ids,
            attention_mask,
        )

    expected_paligemma = model.paligemma_norm(expected_paligemma)
    expected_action = model.action_expert_norm(expected_action)

    assert torch.allclose(
        actual_paligemma,
        expected_paligemma,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        actual_action,
        expected_action,
        atol=1e-6,
        rtol=1e-6,
    )


def test_action_loss_reaches_both_input_streams() -> None:
    model = JointTransformer(TINY_PI0)

    (
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    ) = make_inputs()

    paligemma_hidden_states.requires_grad_(True)
    action_hidden_states.requires_grad_(True)

    _, action_output = model(
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    )

    loss = action_output.square().mean()
    loss.backward()

    assert paligemma_hidden_states.grad is not None
    assert action_hidden_states.grad is not None

    assert paligemma_hidden_states.grad.abs().sum().item() > 0
    assert action_hidden_states.grad.abs().sum().item() > 0


def test_all_layers_receive_gradients() -> None:
    model = JointTransformer(TINY_PI0)

    (
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    ) = make_inputs()

    _, action_output = model(
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    )

    action_output.square().mean().backward()

    for layer in model.layers:
        assert layer.paligemma_layer.self_attn.k_proj.weight.grad is not None
        assert layer.paligemma_layer.self_attn.v_proj.weight.grad is not None
        assert layer.action_expert_layer.self_attn.q_proj.weight.grad is not None
        assert layer.action_expert_layer.mlp.down_proj.weight.grad is not None


def test_final_norm_parameters_receive_gradients() -> None:
    model = JointTransformer(TINY_PI0)

    (
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    ) = make_inputs()

    paligemma_output, action_output = model(
        paligemma_hidden_states,
        action_hidden_states,
        position_ids,
        attention_mask,
    )

    loss = paligemma_output.square().mean() + action_output.square().mean()
    loss.backward()

    assert model.paligemma_norm.weight.grad is not None
    assert model.action_expert_norm.weight.grad is not None


def test_cached_transformer_matches_joint_forward() -> None:
    torch.manual_seed(0)

    model = JointTransformer(TINY_PI0)
    model.eval()

    batch_size = 2
    prefix_length = 5
    suffix_length = 3
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
    block_masks[:, prefix_length] = True
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

    joint_prefix, joint_suffix = model(
        prefix,
        suffix,
        position_ids,
        attention_mask,
    )

    cached_prefix, prefix_cache = model.prefill_prefix(
        prefix_hidden_states=prefix,
        position_ids=position_ids[:, :prefix_length],
        attention_mask=attention_mask[
            :,
            :prefix_length,
            :prefix_length,
        ],
    )
    cached_suffix = model.forward_suffix_with_cache(
        suffix_hidden_states=suffix,
        prefix_cache=prefix_cache,
        position_ids=position_ids[:, prefix_length:],
        attention_mask=attention_mask[:, prefix_length:, :],
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

    assert len(prefix_cache) == TINY_PI0.paligemma.depth

    for prefix_key, prefix_value in prefix_cache:
        assert prefix_key.shape == (
            batch_size,
            TINY_PI0.paligemma.num_kv_heads,
            prefix_length,
            TINY_PI0.paligemma.head_dim,
        )
        assert prefix_value.shape == prefix_key.shape
