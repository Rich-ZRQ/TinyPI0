import pytest
import torch

from configs import TINY_PI0
from pi0.types import IMAGE_KEYS, Observation, validate_actions


def make_observation(batch_size: int = 2) -> Observation:
    image_shape = (
        batch_size,
        TINY_PI0.vision.num_channels,
        TINY_PI0.vision.image_size,
        TINY_PI0.vision.image_size,
    )

    return Observation(
        images={key: torch.zeros(image_shape) for key in IMAGE_KEYS},
        image_masks={key: torch.ones(batch_size, dtype=torch.bool) for key in IMAGE_KEYS},
        state=torch.zeros(
            batch_size,
            TINY_PI0.action_dim,
        ),
        tokenized_prompt=torch.zeros(
            batch_size,
            TINY_PI0.max_token_len,
            dtype=torch.long,
        ),
        tokenized_prompt_mask=torch.ones(
            batch_size,
            TINY_PI0.max_token_len,
            dtype=torch.bool,
        ),
    )


def test_valid_observation() -> None:
    observation = make_observation()
    observation.validate(TINY_PI0)

    assert observation.batch_size == 2
    assert observation.device.type == "cpu"


def test_missing_camera_is_rejected() -> None:
    observation = make_observation()
    observation.images.pop("right_wrist_0_rgb")

    with pytest.raises(ValueError, match="exactly"):
        observation.validate(TINY_PI0)


def test_wrong_image_layout_is_rejected() -> None:
    observation = make_observation()
    observation.images["base_0_rgb"] = torch.zeros(
        2,
        224,
        224,
        3,
    )

    with pytest.raises(ValueError, match="base_0_rgb"):
        observation.validate(TINY_PI0)


def test_prompt_and_mask_must_appear_together() -> None:
    observation = make_observation()
    observation = Observation(
        images=observation.images,
        image_masks=observation.image_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=None,
    )

    with pytest.raises(ValueError, match="provided together"):
        observation.validate(TINY_PI0)


def test_actions_contract() -> None:
    actions = torch.zeros(
        2,
        TINY_PI0.action_horizon,
        TINY_PI0.action_dim,
    )

    validate_actions(
        actions,
        config=TINY_PI0,
        batch_size=2,
    )


def test_wrong_actions_shape_is_rejected() -> None:
    actions = torch.zeros(
        2,
        10,
        TINY_PI0.action_dim,
    )

    with pytest.raises(ValueError, match="actions must have shape"):
        validate_actions(
            actions,
            config=TINY_PI0,
            batch_size=2,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_move_observation_to_cuda() -> None:
    observation = make_observation(batch_size=1)
    cuda_observation = observation.to("cuda")

    assert cuda_observation.device.type == "cuda"
    cuda_observation.validate(TINY_PI0)
