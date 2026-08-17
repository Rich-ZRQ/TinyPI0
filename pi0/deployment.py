"""Load inference-only Tiny pi0 deployment artifacts."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from configs.schema import Pi0Config, TransformerConfig, VisionConfig
from pi0.normalization import NormStats, Pi0Normalizer
from pi0.paligemma_prefix import PaliGemmaPrefixEncoder
from pi0.policy import Pi0Policy


@dataclass(frozen=True)
class DeployMetadata:
    """Validated metadata stored beside deployment weights."""

    step: int
    model_config: Pi0Config
    training_config: dict[str, Any]


def pi0_config_from_dict(raw: Mapping[str, Any]) -> Pi0Config:
    """Reconstruct the nested model configuration saved as JSON."""

    required = {
        "vision",
        "paligemma",
        "action_expert",
        "vocab_size",
        "action_dim",
        "action_horizon",
        "max_token_len",
        "dtype",
    }
    missing = required - set(raw)
    unexpected = set(raw) - required

    if missing or unexpected:
        raise ValueError(f"Invalid model_config keys: missing={sorted(missing)}, unexpected={sorted(unexpected)}")

    try:
        return Pi0Config(
            vision=VisionConfig(**raw["vision"]),
            paligemma=TransformerConfig(**raw["paligemma"]),
            action_expert=TransformerConfig(**raw["action_expert"]),
            vocab_size=int(raw["vocab_size"]),
            action_dim=int(raw["action_dim"]),
            action_horizon=int(raw["action_horizon"]),
            max_token_len=int(raw["max_token_len"]),
            dtype=raw["dtype"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid model_config: {error}") from error


def load_deploy_metadata(artifact_dir: str | Path) -> DeployMetadata:
    """Read and validate deployment metadata without constructing the model."""

    path = Path(artifact_dir) / "metadata.json"

    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid metadata JSON in {path}: {error}") from error

    if not isinstance(raw, dict):
        raise TypeError("metadata.json must contain a JSON object")

    for key in ("step", "model_config", "training_config"):
        if key not in raw:
            raise ValueError(f"metadata.json is missing {key!r}")

    if not isinstance(raw["model_config"], dict):
        raise TypeError("metadata.model_config must be an object")
    if not isinstance(raw["training_config"], dict):
        raise TypeError("metadata.training_config must be an object")

    step = int(raw["step"])

    if step < 0:
        raise ValueError(f"metadata step cannot be negative, got {step}")

    return DeployMetadata(
        step=step,
        model_config=pi0_config_from_dict(raw["model_config"]),
        training_config=dict(raw["training_config"]),
    )


def load_deploy_normalizer(
    artifact_dir: str | Path,
    *,
    use_quantiles: bool = True,
) -> Pi0Normalizer:
    """Reconstruct the normalizer directly from its safetensors state."""

    path = Path(artifact_dir) / "normalizer.safetensors"

    if not path.is_file():
        raise FileNotFoundError(path)

    state = load_file(path, device="cpu")
    required = {
        "state_mean",
        "state_std",
        "action_mean",
        "action_std",
    }

    if use_quantiles:
        required.update(
            {
                "state_q01",
                "state_q99",
                "action_q01",
                "action_q99",
            }
        )

    missing = required - set(state)

    if missing:
        raise ValueError(f"Normalizer state is missing tensors: {sorted(missing)}")

    allowed = required | {
        "state_q01",
        "state_q99",
        "action_q01",
        "action_q99",
    }
    unexpected = set(state) - allowed

    if unexpected:
        raise ValueError(f"Normalizer state has unexpected tensors: {sorted(unexpected)}")

    def stats(prefix: str) -> NormStats:
        return NormStats(
            mean=state[f"{prefix}_mean"].to(torch.float32),
            std=state[f"{prefix}_std"].to(torch.float32),
            q01=(None if f"{prefix}_q01" not in state else state[f"{prefix}_q01"].to(torch.float32)),
            q99=(None if f"{prefix}_q99" not in state else state[f"{prefix}_q99"].to(torch.float32)),
        )

    normalizer = Pi0Normalizer(
        state_stats=stats("state"),
        action_stats=stats("action"),
        use_quantiles=use_quantiles,
    )
    normalizer.load_state_dict(state, strict=True)
    return normalizer


def load_deploy_weights(
    policy: Pi0Policy,
    artifact_dir: str | Path,
    *,
    device: torch.device,
) -> DeployMetadata:
    """Strictly restore learned parameters into an already-built policy."""

    root = Path(artifact_dir)
    metadata = load_deploy_metadata(root)

    if policy.config != metadata.model_config:
        raise ValueError("Policy configuration does not match artifact metadata")

    weights_path = root / "model.safetensors"

    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)

    # Keep the source state on CPU. A new FP32 training artifact is roughly
    # twice the size of its BF16 inference model, so loading both copies onto a
    # 4 GB deployment GPU at once can otherwise cause a transient OOM.
    weights = load_file(weights_path, device="cpu")
    expected = {name for name, parameter in policy.named_parameters() if parameter.requires_grad}
    actual = set(weights)

    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"Artifact parameter mismatch: missing={missing}, unexpected={unexpected}")

    policy.load_state_dict(weights, strict=False)
    policy.eval()
    return metadata


def load_deploy_policy(
    *,
    artifact_dir: str | Path,
    paligemma_snapshot: str | Path,
    device: torch.device,
    use_quantiles: bool = True,
) -> tuple[Pi0Policy, DeployMetadata]:
    """Construct a complete frozen-frontend policy from a deployment directory."""

    metadata = load_deploy_metadata(artifact_dir)
    model_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[metadata.model_config.dtype]
    prefix_encoder = PaliGemmaPrefixEncoder(
        snapshot_path=paligemma_snapshot,
        device=device,
        dtype=model_dtype,
    )
    normalizer = load_deploy_normalizer(
        artifact_dir,
        use_quantiles=use_quantiles,
    )
    policy = Pi0Policy(
        config=metadata.model_config,
        prefix_encoder=prefix_encoder,
        normalizer=normalizer,
        # Inference does not update parameters, so compact BF16 weights are safe.
        trainable_dtype=model_dtype,
    )
    restored_metadata = load_deploy_weights(
        policy,
        artifact_dir,
        device=device,
    )
    return policy, restored_metadata


def find_paligemma_snapshot(explicit_path: str | Path | None = None) -> Path:
    """Resolve an explicit or locally cached PaliGemma 2 snapshot."""

    if explicit_path is not None:
        path = Path(explicit_path).expanduser()

        if not path.is_dir():
            raise FileNotFoundError(path)

        return path

    root = Path.home() / ".cache/huggingface/hub/models--google--paligemma2-3b-pt-224/snapshots"
    snapshots = sorted(path for path in root.glob("*") if path.is_dir())

    if not snapshots:
        raise FileNotFoundError(f"No PaliGemma snapshot found under {root}; pass an explicit snapshot path")

    return snapshots[-1]
