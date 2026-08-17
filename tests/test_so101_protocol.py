import numpy as np
import pytest

from pi0.so101_protocol import (
    MOTOR_POSITION_KEYS,
    check_action_target,
    decode_rgb_image,
    encode_rgb_image,
    make_inference_request,
    robot_action_dict,
    state_from_observation,
    validate_inference_response,
)


def test_image_protocol_round_trip() -> None:
    image = np.full((48, 64, 3), [20, 100, 220], dtype=np.uint8)
    decoded = np.asarray(decode_rgb_image(encode_rgb_image(image, quality=95)))

    assert decoded.shape == image.shape
    assert np.abs(decoded.astype(np.int16) - image.astype(np.int16)).mean() < 3


def test_request_and_response_contract() -> None:
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    request = make_inference_request(
        front_image=image,
        wrist_image=image,
        state=[0, 1, 2, 3, 4, 5],
        prompt="pick up chocolate",
        num_steps=10,
        seed=7,
    )
    response = validate_inference_response(
        {
            "actions": [[0, 1, 2, 3, 4, 5]],
            "training_action_lower": [-10] * 6,
            "training_action_upper": [10] * 6,
        }
    )

    assert set(request["images"]) == {"front", "wrist"}
    assert response["actions"][0] == [0, 1, 2, 3, 4, 5]


def test_safety_gate_rejects_large_delta_and_range_violation() -> None:
    reasons = check_action_target(
        target=[30, 0, 0, 0, 0, 0],
        current=[0] * 6,
        lower=[-20] * 6,
        upper=[20] * 6,
        reject_delta=25,
        range_margin=5,
    )

    assert any("requested delta" in reason for reason in reasons)
    assert any("outside training range" in reason for reason in reasons)


def test_safety_gate_accepts_conservative_target() -> None:
    assert not check_action_target(
        target=[1, 2, 3, 4, 5, 6],
        current=[0, 1, 2, 3, 4, 5],
        lower=[-10] * 6,
        upper=[10] * 6,
        reject_delta=5,
        range_margin=0,
    )


def test_lerobot_joint_order_mapping() -> None:
    observation = {key: float(index) for index, key in enumerate(MOTOR_POSITION_KEYS)}

    assert state_from_observation(observation) == [0, 1, 2, 3, 4, 5]
    assert robot_action_dict([10, 11, 12, 13, 14, 15]) == {
        key: float(index + 10) for index, key in enumerate(MOTOR_POSITION_KEYS)
    }


def test_protocol_rejects_non_finite_action() -> None:
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_inference_response(
            {
                "actions": [[0, 1, 2, 3, 4, float("nan")]],
                "training_action_lower": [-10] * 6,
                "training_action_upper": [10] * 6,
            }
        )
