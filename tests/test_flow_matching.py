import pytest
import torch

from configs import TINY_PI0
from pi0.flow_matching import (
    flow_matching_loss,
    make_flow_matching_target,
    sample_noise,
    sample_time,
)


def make_actions(
    *,
    batch_size: int = 4,
) -> torch.Tensor:
    return torch.randn(
        batch_size,
        TINY_PI0.action_horizon,
        TINY_PI0.action_dim,
    )


def test_sample_noise_contract() -> None:
    shape = (
        4,
        TINY_PI0.action_horizon,
        TINY_PI0.action_dim,
    )

    noise = sample_noise(
        shape,
        device="cpu",
    )

    assert noise.shape == shape
    assert noise.dtype == torch.float32
    assert noise.device.type == "cpu"
    assert noise.is_floating_point()


def test_sample_time_contract() -> None:
    torch.manual_seed(0)

    timestep = sample_time(
        1024,
        device="cpu",
    )

    assert timestep.shape == (1024,)
    assert timestep.dtype == torch.float32
    assert timestep.device.type == "cpu"

    assert torch.all(timestep >= 0.001)
    assert torch.all(timestep <= 1.0)


def test_timestep_zero_returns_clean_actions() -> None:
    actions = make_actions()
    noise = torch.randn_like(actions)
    timestep = torch.zeros(actions.shape[0])

    noisy_actions, target_velocity = make_flow_matching_target(
        actions,
        noise,
        timestep,
    )

    assert torch.equal(
        noisy_actions,
        actions,
    )
    assert torch.equal(
        target_velocity,
        noise - actions,
    )


def test_timestep_one_returns_noise() -> None:
    actions = make_actions()
    noise = torch.randn_like(actions)
    timestep = torch.ones(actions.shape[0])

    noisy_actions, _ = make_flow_matching_target(
        actions,
        noise,
        timestep,
    )

    assert torch.equal(
        noisy_actions,
        noise,
    )


def test_each_batch_item_uses_its_own_timestep() -> None:
    actions = torch.zeros(
        2,
        TINY_PI0.action_horizon,
        TINY_PI0.action_dim,
    )
    noise = torch.ones_like(actions)

    timestep = torch.tensor([0.25, 0.75])

    noisy_actions, target_velocity = make_flow_matching_target(
        actions,
        noise,
        timestep,
    )

    assert torch.allclose(
        noisy_actions[0],
        torch.full_like(noisy_actions[0], 0.25),
    )
    assert torch.allclose(
        noisy_actions[1],
        torch.full_like(noisy_actions[1], 0.75),
    )
    assert torch.equal(
        target_velocity,
        torch.ones_like(target_velocity),
    )


def test_loss_matches_elementwise_squared_error() -> None:
    predicted_velocity = torch.tensor([[[1.0, 3.0], [2.0, -1.0]]])
    target_velocity = torch.tensor([[[0.0, 1.0], [4.0, -1.0]]])

    loss = flow_matching_loss(
        predicted_velocity,
        target_velocity,
    )

    expected = (predicted_velocity - target_velocity).square()

    assert loss.shape == predicted_velocity.shape
    assert torch.equal(loss, expected)


def test_loss_backpropagates_to_prediction() -> None:
    predicted_velocity = torch.randn(
        2,
        TINY_PI0.action_horizon,
        TINY_PI0.action_dim,
        requires_grad=True,
    )
    target_velocity = torch.randn_like(predicted_velocity)

    loss = flow_matching_loss(
        predicted_velocity,
        target_velocity,
    ).mean()

    loss.backward()

    assert predicted_velocity.grad is not None
    assert torch.isfinite(predicted_velocity.grad).all()
    assert predicted_velocity.grad.abs().sum().item() > 0


@pytest.mark.parametrize(
    ("noise_shape", "time_shape"),
    [
        (
            (
                2,
                TINY_PI0.action_horizon - 1,
                TINY_PI0.action_dim,
            ),
            (2,),
        ),
        (
            (
                2,
                TINY_PI0.action_horizon,
                TINY_PI0.action_dim,
            ),
            (2, 1),
        ),
    ],
)
def test_invalid_flow_shapes_are_rejected(
    noise_shape,
    time_shape,
) -> None:
    actions = make_actions(batch_size=2)
    noise = torch.randn(noise_shape)
    timestep = torch.randn(time_shape)

    with pytest.raises(ValueError):
        make_flow_matching_target(
            actions,
            noise,
            timestep,
        )
