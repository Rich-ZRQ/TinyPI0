"""Serve the trained Tiny pi0 policy to a separate LeRobot process."""

import json
import math
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
import tyro

from pi0.deployment import find_paligemma_snapshot, load_deploy_policy
from pi0.processor import Pi0Processor
from pi0.so101_protocol import decode_rgb_image
from pi0.types import IMAGE_KEYS

MAX_REQUEST_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class Args:
    artifact_dir: Path = Path("artifacts/pi0_so101_recommended_step7000")
    paligemma_snapshot: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8000


class So101PolicyRuntime:
    """Own the GPU policy and serialize inference calls."""

    def __init__(self, args: Args) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the SO101 policy server")

        self.device = torch.device("cuda")
        self.snapshot = find_paligemma_snapshot(args.paligemma_snapshot)
        self.policy, self.metadata = load_deploy_policy(
            artifact_dir=args.artifact_dir,
            paligemma_snapshot=self.snapshot,
            device=self.device,
            use_quantiles=True,
        )
        self.processor = Pi0Processor(
            config=self.metadata.model_config,
            snapshot_path=self.snapshot,
        )
        self.lock = threading.Lock()
        self.robot_action_dim = self.policy.normalizer.action_mean.shape[0]

        if self.robot_action_dim != 6:
            raise ValueError(f"Expected a six-dimensional SO101 normalizer, got {self.robot_action_dim}")

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        images = payload.get("images")

        if not isinstance(images, dict):
            raise TypeError("Request 'images' must be an object")
        if set(images) != {"front", "wrist"}:
            raise ValueError("Request images must contain exactly 'front' and 'wrist'")

        state = payload.get("state")

        if not isinstance(state, list) or len(state) != self.robot_action_dim:
            raise ValueError(f"Request state must contain {self.robot_action_dim} values")
        if not all(isinstance(value, int | float) for value in state):
            raise TypeError("Request state values must be numeric")
        if not all(math.isfinite(value) for value in state):
            raise ValueError("Request state contains NaN or Inf")

        prompt = payload.get("prompt")

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Request prompt must be a non-empty string")

        num_steps = int(payload.get("num_steps", 10))
        seed = int(payload.get("seed", 0))

        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if seed < 0:
            raise ValueError(f"seed cannot be negative, got {seed}")

        padded_state = torch.zeros(
            1,
            self.metadata.model_config.action_dim,
            dtype=torch.float32,
        )
        padded_state[0, : self.robot_action_dim] = torch.tensor(state, dtype=torch.float32)
        observation = self.processor(
            images={
                IMAGE_KEYS[0]: [decode_rgb_image(images["front"])],
                IMAGE_KEYS[1]: [decode_rgb_image(images["wrist"])],
            },
            prompts=[prompt],
            state=padded_state,
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(
            1,
            self.metadata.model_config.action_horizon,
            self.metadata.model_config.action_dim,
            device=self.device,
            dtype=self.policy.model_dtype,
            generator=generator,
        )

        with self.lock:
            torch.cuda.synchronize()
            started_at = time.perf_counter()
            actions = self.policy.sample_actions(
                observation,
                num_steps=num_steps,
                noise=noise,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started_at

        if not torch.isfinite(actions).all():
            raise RuntimeError("Policy produced NaN or Inf actions")

        robot_actions = actions[0, :, : self.robot_action_dim].cpu()
        return {
            "artifact_step": self.metadata.step,
            "actions": robot_actions.tolist(),
            "training_action_lower": self.policy.normalizer.action_q01.cpu().tolist(),
            "training_action_upper": self.policy.normalizer.action_q99.cpu().tolist(),
            "inference_seconds": elapsed,
            "num_steps": num_steps,
            "seed": seed,
        }


def make_handler(runtime: So101PolicyRuntime) -> type[BaseHTTPRequestHandler]:
    class PolicyRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return

            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ready",
                    "artifact_step": runtime.metadata.step,
                    "device": str(runtime.device),
                },
            )

        def do_POST(self) -> None:
            if self.path != "/v1/infer":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))

                if not 0 < content_length <= MAX_REQUEST_BYTES:
                    raise ValueError(f"Invalid request size: {content_length}")

                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))

                if not isinstance(payload, dict):
                    raise TypeError("Request body must be a JSON object")

                response = runtime.infer(payload)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            except Exception as error:  # noqa: BLE001
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})
                return

            self._send_json(HTTPStatus.OK, response)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[{self.log_date_time_string()}] {format % args}")

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return PolicyRequestHandler


def main(args: Args) -> None:
    if not 1 <= args.port <= 65535:
        raise ValueError(f"port must be between 1 and 65535, got {args.port}")

    runtime = So101PolicyRuntime(args)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(runtime),
    )
    print(f"TinyPi0 policy ready: http://{args.host}:{args.port}")
    print(f"Artifact step: {runtime.metadata.step}")
    print("Endpoints: GET /health, POST /v1/infer")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping policy server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main(tyro.cli(Args))
