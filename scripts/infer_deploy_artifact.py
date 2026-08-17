"""Run a trained deployment artifact on one recorded SO101 observation."""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

from pi0.deployment import (
    find_paligemma_snapshot,
    load_deploy_metadata,
    load_deploy_policy,
)
from pi0.lerobot_dataset import LeRobotPi0Dataset
from pi0.processor import Pi0Processor
from pi0.types import IMAGE_KEYS


@dataclass(frozen=True)
class Args:
    artifact_dir: Path = Path("artifacts/pi0_so101_recommended_step10000")
    dataset_root: Path = Path.home() / ".cache/huggingface/lerobot/Rich-RZ/so101_chocolates_to_bowl_v1"
    paligemma_snapshot: Path | None = None
    sample_index: int = 0
    num_steps: int = 10
    seed: int = 0
    robot_action_dim: int = 6
    max_first_action_delta: float = 10.0
    output_json: Path | None = None


def main(args: Args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SO101 deployment profile")
    if args.num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {args.num_steps}")
    if args.robot_action_dim <= 0:
        raise ValueError(f"robot_action_dim must be positive, got {args.robot_action_dim}")
    if args.max_first_action_delta <= 0:
        raise ValueError(f"max_first_action_delta must be positive, got {args.max_first_action_delta}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda")
    metadata = load_deploy_metadata(args.artifact_dir)

    if args.robot_action_dim > metadata.model_config.action_dim:
        raise ValueError(
            f"robot_action_dim={args.robot_action_dim} exceeds model action_dim={metadata.model_config.action_dim}"
        )
    if not 0 <= args.sample_index:
        raise ValueError(f"sample_index cannot be negative, got {args.sample_index}")

    snapshot = find_paligemma_snapshot(args.paligemma_snapshot)
    dataset = LeRobotPi0Dataset(
        root=args.dataset_root,
        config=metadata.model_config,
    )

    if args.sample_index >= len(dataset):
        raise IndexError(f"sample_index={args.sample_index} is outside dataset length {len(dataset)}")

    sample = dataset[args.sample_index]
    processor = Pi0Processor(
        config=metadata.model_config,
        snapshot_path=snapshot,
    )
    observation = processor(
        images={key: [sample.images[key]] for key in IMAGE_KEYS},
        prompts=[sample.prompt],
        state=sample.state.unsqueeze(0),
    )
    policy, metadata = load_deploy_policy(
        artifact_dir=args.artifact_dir,
        paligemma_snapshot=snapshot,
        device=device,
        use_quantiles=True,
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(
        1,
        metadata.model_config.action_horizon,
        metadata.model_config.action_dim,
        device=device,
        dtype=policy.model_dtype,
        generator=generator,
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started_at = time.perf_counter()
    actions = policy.sample_actions(
        observation,
        num_steps=args.num_steps,
        noise=noise,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started_at

    if actions.shape != (
        1,
        metadata.model_config.action_horizon,
        metadata.model_config.action_dim,
    ):
        raise RuntimeError(f"Unexpected action shape: {tuple(actions.shape)}")
    if not torch.isfinite(actions).all():
        raise RuntimeError("Model produced NaN or Inf actions")

    robot_actions = actions[0, :, : args.robot_action_dim].cpu()
    current_state = sample.state[: args.robot_action_dim].cpu()
    recorded_actions = sample.actions[:, : args.robot_action_dim].cpu()
    valid_steps = sample.action_valid_mask.cpu()
    first_delta = robot_actions[0] - current_state
    first_target_error = robot_actions[0] - recorded_actions[0]
    valid_absolute_error = (robot_actions[valid_steps] - recorded_actions[valid_steps]).abs()
    q01 = policy.normalizer.action_q01[: args.robot_action_dim].cpu()
    q99 = policy.normalizer.action_q99[: args.robot_action_dim].cpu()
    in_training_range = (robot_actions >= q01) & (robot_actions <= q99)
    max_abs_first_delta = first_delta.abs().max().item()
    passes_first_delta_gate = max_abs_first_delta <= args.max_first_action_delta

    report = {
        "artifact_step": metadata.step,
        "sample_index": args.sample_index,
        "episode_index": sample.episode_index,
        "frame_index": sample.frame_index,
        "timestamp": sample.timestamp,
        "prompt": sample.prompt,
        "dataset_fps": dataset.fps,
        "num_flow_steps": args.num_steps,
        "elapsed_seconds": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "action_shape": list(actions.shape),
        "robot_action_dim": args.robot_action_dim,
        "current_state": current_state.tolist(),
        "recorded_first_action": recorded_actions[0].tolist(),
        "first_action": robot_actions[0].tolist(),
        "first_action_delta": first_delta.tolist(),
        "first_action_error": first_target_error.tolist(),
        "valid_chunk_mae": valid_absolute_error.mean().item(),
        "max_abs_first_action_delta": max_abs_first_delta,
        "max_first_action_delta_limit": args.max_first_action_delta,
        "passes_first_action_delta_gate": passes_first_delta_gate,
        "predicted_min": robot_actions.amin(dim=0).tolist(),
        "predicted_max": robot_actions.amax(dim=0).tolist(),
        "training_q01": q01.tolist(),
        "training_q99": q99.tolist(),
        "fraction_inside_training_range": in_training_range.to(torch.float32).mean().item(),
        "ready_for_hardware": passes_first_delta_gate,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    **report,
                    "robot_actions": robot_actions.tolist(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Saved full action chunk to {args.output_json}")

    if not passes_first_delta_gate:
        raise RuntimeError(
            "Offline safety gate failed: first predicted action is too far from the recorded current state; "
            "do not connect this checkpoint to SO101"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
