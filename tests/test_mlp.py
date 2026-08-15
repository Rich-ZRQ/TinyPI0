import pytest
import torch
from torch.nn import functional as F

from configs import TINY_PI0
from pi0.mlp import GemmaMLP


@pytest.mark.parametrize(
    "config",
    [
        TINY_PI0.paligemma,
        TINY_PI0.action_expert,
    ],
)
def test_mlp_projection_shapes(config) -> None:
    mlp = GemmaMLP(config)

    assert mlp.gate_proj.weight.shape == (
        config.mlp_dim,
        config.width,
    )
    assert mlp.up_proj.weight.shape == (
        config.mlp_dim,
        config.width,
    )
    assert mlp.down_proj.weight.shape == (
        config.width,
        config.mlp_dim,
    )

    assert mlp.gate_proj.bias is None
    assert mlp.up_proj.bias is None
    assert mlp.down_proj.bias is None


@pytest.mark.parametrize(
    "config",
    [
        TINY_PI0.paligemma,
        TINY_PI0.action_expert,
    ],
)
def test_mlp_preserves_external_shape(config) -> None:
    mlp = GemmaMLP(config)

    hidden_states = torch.randn(
        2,
        5,
        config.width,
    )

    output = mlp(hidden_states)

    assert output.shape == hidden_states.shape


def test_mlp_matches_explicit_formula() -> None:
    config = TINY_PI0.action_expert
    mlp = GemmaMLP(config)

    hidden_states = torch.randn(
        2,
        4,
        config.width,
    )

    actual = mlp(hidden_states)

    gate = F.gelu(
        mlp.gate_proj(hidden_states),
        approximate="tanh",
    )
    up = mlp.up_proj(hidden_states)
    expected = mlp.down_proj(gate * up)

    assert torch.allclose(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_gradient_flows_through_all_projections() -> None:
    config = TINY_PI0.action_expert
    mlp = GemmaMLP(config)

    hidden_states = torch.randn(
        2,
        5,
        config.width,
        requires_grad=True,
    )

    output = mlp(hidden_states)
    loss = output.square().mean()
    loss.backward()

    assert hidden_states.grad is not None
    assert mlp.gate_proj.weight.grad is not None
    assert mlp.up_proj.weight.grad is not None
    assert mlp.down_proj.weight.grad is not None

    assert torch.isfinite(hidden_states.grad).all()
    assert torch.isfinite(mlp.gate_proj.weight.grad).all()
    assert torch.isfinite(mlp.up_proj.weight.grad).all()
    assert torch.isfinite(mlp.down_proj.weight.grad).all()


def test_wrong_hidden_width_is_rejected() -> None:
    config = TINY_PI0.action_expert
    mlp = GemmaMLP(config)

    hidden_states = torch.randn(
        2,
        5,
        config.width + 1,
    )

    with pytest.raises(
        ValueError,
        match="expected hidden width",
    ):
        mlp(hidden_states)


def test_non_floating_input_is_rejected() -> None:
    config = TINY_PI0.action_expert
    mlp = GemmaMLP(config)

    hidden_states = torch.ones(
        2,
        5,
        config.width,
        dtype=torch.long,
    )

    with pytest.raises(
        TypeError,
        match="must be floating point",
    ):
        mlp(hidden_states)
