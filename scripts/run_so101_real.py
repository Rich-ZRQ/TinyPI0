"""Run the Tiny pi0 policy against a LeRobot SO101 with safety gates."""

import argparse
import time
from pathlib import Path

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from pi0.so101_protocol import (
    MOTOR_POSITION_KEYS,
    check_action_target,
    get_json,
    make_inference_request,
    post_json,
    robot_action_dict,
    state_from_observation,
    validate_inference_response,
)

DEFAULT_PROMPT = "Pick up all chocolates from the table and place them into the bowl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-port", required=True, help="SO101 serial port, preferably /dev/serial/by-id/...")
    parser.add_argument("--robot-id", default="my_follower", help="LeRobot calibration id")
    parser.add_argument("--front-camera", default="0", help="Front OpenCV index or /dev/video path")
    parser.add_argument("--wrist-camera", default="1", help="Wrist OpenCV index or /dev/video path")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--control-fps", type=int, default=20)
    parser.add_argument("--front-camera-fps", type=int, default=30)
    parser.add_argument("--wrist-camera-fps", type=int, default=60)
    parser.add_argument("--front-camera-fourcc", default="MJPG")
    parser.add_argument("--wrist-camera-fourcc", default="MJPG")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seed-mode",
        choices=("increment", "fixed"),
        default="increment",
        help="Use seed+cycle for each replan, or reuse one fixed seed",
    )
    parser.add_argument("--max-cycles", type=int, default=10, help="Zero means run until interrupted")
    parser.add_argument("--actions-per-inference", type=int, default=1)
    parser.add_argument("--max-relative-target", type=float, default=2.0)
    parser.add_argument("--reject-target-delta", type=float, default=10.0)
    parser.add_argument("--training-range-margin", type=float, default=5.0)
    parser.add_argument(
        "--lerobot-safety-only",
        action="store_true",
        help="Disable policy delta/range rejection and rely on LeRobot's limits",
    )
    parser.add_argument("--request-timeout", type=float, default=20.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--execute", action="store_true", help="Actually send actions; default is dry-run")
    parser.add_argument("--yes", action="store_true", help="Skip the interactive execute confirmation")
    return parser.parse_args()


def camera_source(value: str) -> int | Path:
    try:
        return int(value)
    except ValueError:
        return Path(value)


def require_device_path(value: str, *, name: str) -> None:
    """Fail before touching the robot when an explicit Linux device vanished."""

    try:
        int(value)
    except ValueError:
        path = Path(value)

        if not path.exists():
            raise FileNotFoundError(
                f"{name} device does not exist: {path}. "
                "Check usbipd list in Windows and re-attach the USB device to WSL."
            ) from None


def disconnect_partial_robot(robot: SO101Follower) -> None:
    """Clean up even when SO101Follower.connect() failed halfway through."""

    cleanup_errors: list[str] = []

    if robot.bus.is_connected:
        try:
            robot.bus.disconnect(robot.config.disable_torque_on_disconnect)
        except Exception as error:  # noqa: BLE001
            cleanup_errors.append(f"motor bus: {error}")

    for name, camera in robot.cameras.items():
        if camera.is_connected:
            try:
                camera.disconnect()
            except Exception as error:  # noqa: BLE001
                cleanup_errors.append(f"camera {name}: {error}")

    if cleanup_errors:
        print("Cleanup warnings: " + "; ".join(cleanup_errors))
    else:
        print("Robot resources released; torque disabled if the bus was connected")


def require_camera_threads(robot: SO101Follower) -> None:
    """Abort an open-loop chunk as soon as either camera reader stops."""

    failed = [name for name, camera in robot.cameras.items() if camera.thread is None or not camera.thread.is_alive()]

    if failed:
        raise RuntimeError(
            f"Camera read thread stopped during action execution: {failed}. "
            "No more actions will be sent; re-attach the USB camera to WSL."
        )


def validate_args(args: argparse.Namespace) -> None:
    if args.control_fps <= 0:
        raise ValueError("control-fps must be positive")
    if args.front_camera_fps <= 0:
        raise ValueError("front-camera-fps must be positive")
    if args.wrist_camera_fps <= 0:
        raise ValueError("wrist-camera-fps must be positive")
    if len(args.front_camera_fourcc) != 4:
        raise ValueError("front-camera-fourcc must contain four characters")
    if len(args.wrist_camera_fourcc) != 4:
        raise ValueError("wrist-camera-fourcc must contain four characters")
    if args.num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if args.seed < 0:
        raise ValueError("seed cannot be negative")
    if args.max_cycles < 0:
        raise ValueError("max_cycles cannot be negative")
    if not 1 <= args.actions_per_inference <= 50:
        raise ValueError("actions_per_inference must be between 1 and 50")
    if args.max_relative_target <= 0:
        raise ValueError("max_relative_target must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    require_device_path(args.robot_port, name="robot serial")
    require_device_path(args.front_camera, name="front camera")
    require_device_path(args.wrist_camera, name="wrist camera")

    if args.execute and not args.yes:
        confirmation = input("REAL ROBOT EXECUTION ENABLED. Keep an emergency stop ready. Type EXECUTE to continue: ")

        if confirmation != "EXECUTE":
            raise RuntimeError("Execution cancelled")

    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=camera_source(args.front_camera),
            width=640,
            height=480,
            fps=args.front_camera_fps,
            fourcc=args.front_camera_fourcc,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=camera_source(args.wrist_camera),
            width=640,
            height=480,
            fps=args.wrist_camera_fps,
            fourcc=args.wrist_camera_fourcc,
        ),
    }
    robot_config = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras=cameras,
        use_degrees=True,
        disable_torque_on_disconnect=True,
        max_relative_target=args.max_relative_target,
    )
    robot = SO101Follower(robot_config)
    health_url = args.server_url.rstrip("/") + "/health"
    infer_url = args.server_url.rstrip("/") + "/v1/infer"
    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"Mode: {mode}")
    print(f"Policy: {infer_url}")
    print(f"Calibration id: {args.robot_id}")
    print(f"Per-command LeRobot clipping: {args.max_relative_target}")
    print("Safety profile: " + ("LEROBOT-ONLY" if args.lerobot_safety_only else "POLICY + LEROBOT"))

    health = get_json(health_url, timeout=args.request_timeout)

    if health.get("status") != "ready":
        raise RuntimeError(f"Policy server is not ready: {health}")

    print(f"Policy server ready: step={health.get('artifact_step')} device={health.get('device')}")

    try:
        robot.connect()
        cycle = 0

        while args.max_cycles == 0 or cycle < args.max_cycles:
            observation = robot.get_observation()
            state = state_from_observation(observation)
            request_seed = args.seed + cycle if args.seed_mode == "increment" else args.seed
            payload = make_inference_request(
                front_image=observation["front"],
                wrist_image=observation["wrist"],
                state=state,
                prompt=args.prompt,
                num_steps=args.num_steps,
                seed=request_seed,
                jpeg_quality=args.jpeg_quality,
            )
            response = validate_inference_response(
                post_json(
                    infer_url,
                    payload,
                    timeout=args.request_timeout,
                )
            )
            actions = response["actions"][: args.actions_per_inference]
            lower = response["training_action_lower"]
            upper = response["training_action_upper"]
            print(
                f"cycle={cycle} step={response.get('artifact_step')} "
                f"seed={response.get('seed')} "
                f"inference={response.get('inference_seconds', float('nan')):.3f}s "
                f"state={[round(value, 2) for value in state]}"
            )

            reference = state

            for action_index, target in enumerate(actions):
                deltas = [
                    target_value - reference_value
                    for target_value, reference_value in zip(target, reference, strict=True)
                ]
                print(f"  target[{action_index}]={[round(value, 2) for value in target]}")
                print(f"  delta[{action_index}]={[round(value, 2) for value in deltas]}")
                print(
                    "  training-range="
                    f"{[(round(low, 2), round(high, 2)) for low, high in zip(lower, upper, strict=True)]}"
                )
                reasons = []

                if not args.lerobot_safety_only:
                    reasons = check_action_target(
                        target,
                        reference,
                        lower,
                        upper,
                        reject_delta=args.reject_target_delta,
                        range_margin=args.training_range_margin,
                    )

                if reasons:
                    joined_reasons = "\n  - ".join(reasons)

                    if args.execute:
                        raise RuntimeError(f"Safety gate rejected action:\n  - {joined_reasons}")

                    print(f"  DRY-RUN REJECTED:\n  - {joined_reasons}")
                    break

                command = robot_action_dict(target)

                if args.execute:
                    require_camera_threads(robot)
                    sent = robot.send_action(command)
                    reference = [float(sent[key]) for key in MOTOR_POSITION_KEYS]
                    print(f"  sent[{action_index}]={[round(value, 2) for value in reference]}")
                    time.sleep(1.0 / args.control_fps)
                else:
                    print(f"  dry-run[{action_index}]=accepted")

            cycle += 1
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        disconnect_partial_robot(robot)


if __name__ == "__main__":
    main()
