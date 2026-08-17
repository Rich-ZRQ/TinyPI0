"""Wire protocol and safety checks shared by policy and SO101 processes."""

import base64
import json
import math
from collections.abc import Mapping, Sequence
from io import BytesIO
from typing import Any
from urllib import request

import numpy as np
from PIL import Image

MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
MOTOR_POSITION_KEYS = tuple(f"{name}.pos" for name in MOTOR_NAMES)
CAMERA_NAMES = ("front", "wrist")


def encode_rgb_image(image: Image.Image | np.ndarray, *, quality: int = 90) -> str:
    """Encode one RGB observation as a base64 JPEG string."""

    if not 1 <= quality <= 100:
        raise ValueError(f"quality must be between 1 and 100, got {quality}")

    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image array must have shape [H, W, 3], got {image.shape}")
        if image.dtype != np.uint8:
            raise TypeError(f"image array must use uint8, got {image.dtype}")
        pil_image = Image.fromarray(image, mode="RGB")
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        raise TypeError(f"image must be PIL.Image or numpy.ndarray, got {type(image).__name__}")

    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_rgb_image(encoded: str) -> Image.Image:
    """Decode one base64 JPEG into an owned RGB PIL image."""

    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid base64 image") from error

    try:
        with Image.open(BytesIO(raw)) as image:
            return image.convert("RGB").copy()
    except OSError as error:
        raise ValueError("Encoded payload is not a supported image") from error


def make_inference_request(
    *,
    front_image: Image.Image | np.ndarray,
    wrist_image: Image.Image | np.ndarray,
    state: Sequence[float],
    prompt: str,
    num_steps: int,
    seed: int,
    jpeg_quality: int = 90,
) -> dict[str, Any]:
    """Create a JSON-serializable policy request."""

    state_values = _finite_vector(state, name="state", expected_length=len(MOTOR_NAMES))

    if not prompt.strip():
        raise ValueError("prompt cannot be empty")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if seed < 0:
        raise ValueError(f"seed cannot be negative, got {seed}")

    return {
        "images": {
            "front": encode_rgb_image(front_image, quality=jpeg_quality),
            "wrist": encode_rgb_image(wrist_image, quality=jpeg_quality),
        },
        "state": state_values,
        "prompt": prompt,
        "num_steps": num_steps,
        "seed": seed,
    }


def validate_inference_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the policy response before any value reaches robot code."""

    if "actions" not in payload:
        raise ValueError("Policy response is missing 'actions'")

    raw_actions = payload["actions"]

    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError("Policy response actions must be a non-empty list")

    actions = [
        _finite_vector(action, name=f"actions[{index}]", expected_length=len(MOTOR_NAMES))
        for index, action in enumerate(raw_actions)
    ]
    lower = _finite_vector(
        payload.get("training_action_lower"),
        name="training_action_lower",
        expected_length=len(MOTOR_NAMES),
    )
    upper = _finite_vector(
        payload.get("training_action_upper"),
        name="training_action_upper",
        expected_length=len(MOTOR_NAMES),
    )

    if any(low >= high for low, high in zip(lower, upper, strict=True)):
        raise ValueError("Every training action lower bound must be smaller than its upper bound")

    return {
        **dict(payload),
        "actions": actions,
        "training_action_lower": lower,
        "training_action_upper": upper,
    }


def check_action_target(
    target: Sequence[float],
    current: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    reject_delta: float,
    range_margin: float,
) -> list[str]:
    """Return human-readable reasons why an absolute joint target is unsafe."""

    if reject_delta <= 0:
        raise ValueError(f"reject_delta must be positive, got {reject_delta}")
    if range_margin < 0:
        raise ValueError(f"range_margin cannot be negative, got {range_margin}")

    target_values = _finite_vector(target, name="target", expected_length=len(MOTOR_NAMES))
    current_values = _finite_vector(current, name="current", expected_length=len(MOTOR_NAMES))
    lower_values = _finite_vector(lower, name="lower", expected_length=len(MOTOR_NAMES))
    upper_values = _finite_vector(upper, name="upper", expected_length=len(MOTOR_NAMES))
    reasons: list[str] = []

    for name, target_value, current_value, low, high in zip(
        MOTOR_NAMES,
        target_values,
        current_values,
        lower_values,
        upper_values,
        strict=True,
    ):
        delta = abs(target_value - current_value)

        if delta > reject_delta:
            reasons.append(f"{name}: requested delta {delta:.3f} exceeds rejection limit {reject_delta:.3f}")

        if not low - range_margin <= target_value <= high + range_margin:
            reasons.append(
                f"{name}: target {target_value:.3f} is outside training range "
                f"[{low:.3f}, {high:.3f}] with margin {range_margin:.3f}"
            )

    return reasons


def state_from_observation(observation: Mapping[str, Any]) -> list[float]:
    """Extract the six training-ordered joint positions from LeRobot output."""

    missing = [key for key in MOTOR_POSITION_KEYS if key not in observation]

    if missing:
        raise ValueError(f"Robot observation is missing motor positions: {missing}")

    return _finite_vector(
        [observation[key] for key in MOTOR_POSITION_KEYS],
        name="robot state",
        expected_length=len(MOTOR_NAMES),
    )


def robot_action_dict(values: Sequence[float]) -> dict[str, float]:
    """Map a training-ordered six-vector to LeRobot's action dictionary."""

    vector = _finite_vector(values, name="action", expected_length=len(MOTOR_NAMES))
    return dict(zip(MOTOR_POSITION_KEYS, vector, strict=True))


def post_json(url: str, payload: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
    """POST JSON using only the Python standard library."""

    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")

    encoded = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(http_request, timeout=timeout) as response:
        response_payload = json.loads(response.read().decode("utf-8"))

    if not isinstance(response_payload, dict):
        raise TypeError("Policy server response must be a JSON object")

    return response_payload


def get_json(url: str, *, timeout: float) -> dict[str, Any]:
    """GET and decode one JSON object using only the standard library."""

    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")

    with request.urlopen(url, timeout=timeout) as response:
        response_payload = json.loads(response.read().decode("utf-8"))

    if not isinstance(response_payload, dict):
        raise TypeError("Policy server response must be a JSON object")

    return response_payload


def _finite_vector(
    values: Any,
    *,
    name: str,
    expected_length: int,
) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        raise TypeError(f"{name} must be a sequence")
    if len(values) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} values, got {len(values)}")

    vector = [float(value) for value in values]

    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} contains NaN or Inf")

    return vector
