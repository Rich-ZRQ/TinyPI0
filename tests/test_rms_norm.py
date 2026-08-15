import pytest
import torch

from pi0.rms_norm import GemmaRMSNorm


def test_output_matches_manual_formula() -> None:
    layer = GemmaRMSNorm(
        dim=4,
        eps=1e-6,
    )

    x = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [-1.0, 0.5, 2.0, -3.0],
            ]
        ],
        dtype=torch.float32,
    )

    actual = layer(x)

    mean_square = torch.mean(
        torch.square(x),
        dim=-1,
        keepdim=True,
    )
    expected = x * torch.rsqrt(mean_square + 1e-6)

    assert torch.allclose(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_zero_initialized_weight_preserves_normalized_value() -> None:
    layer = GemmaRMSNorm(dim=8)

    assert torch.equal(
        layer.weight,
        torch.zeros_like(layer.weight),
    )

    x = torch.randn(2, 5, 8)
    output = layer(x)

    output_mean_square = torch.mean(
        torch.square(output),
        dim=-1,
    )

    assert torch.allclose(
        output_mean_square,
        torch.ones_like(output_mean_square),
        atol=1e-4,
        rtol=1e-4,
    )


def test_weight_uses_one_plus_parameter_convention() -> None:
    layer = GemmaRMSNorm(dim=4)

    x = torch.randn(2, 3, 4)

    baseline = layer(x)

    with torch.no_grad():
        layer.weight.fill_(0.5)

    scaled = layer(x)

    assert torch.allclose(
        scaled,
        baseline * 1.5,
        atol=1e-6,
        rtol=1e-6,
    )


def test_preserves_input_dtype() -> None:
    layer = GemmaRMSNorm(dim=4)

    x = torch.randn(
        2,
        3,
        4,
        dtype=torch.bfloat16,
    )

    output = layer(x)

    assert output.dtype == torch.bfloat16
    assert output.shape == x.shape


def test_gradient_flows_to_input_and_weight() -> None:
    layer = GemmaRMSNorm(dim=4)

    x = torch.randn(
        2,
        3,
        4,
        requires_grad=True,
    )

    loss = layer(x).square().mean()
    loss.backward()

    assert x.grad is not None
    assert layer.weight.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weight.grad).all()


def test_wrong_last_dimension_is_rejected() -> None:
    layer = GemmaRMSNorm(dim=4)
    x = torch.randn(2, 3, 5)

    with pytest.raises(
        ValueError,
        match="expected last dimension 4",
    ):
        layer(x)


def test_integer_input_is_rejected() -> None:
    layer = GemmaRMSNorm(dim=4)
    x = torch.ones(
        2,
        3,
        4,
        dtype=torch.long,
    )

    with pytest.raises(
        TypeError,
        match="floating point",
    ):
        layer(x)


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="dim must be positive",
    ):
        GemmaRMSNorm(dim=0)

    with pytest.raises(
        ValueError,
        match="eps must be positive",
    ):
        GemmaRMSNorm(dim=4, eps=0.0)
