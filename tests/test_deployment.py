import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import Tensor, nn

from configs import TINY_PI0
from pi0.deployment import (
    load_deploy_metadata,
    load_deploy_normalizer,
    load_deploy_weights,
)
from pi0.normalization import NormStats, Pi0Normalizer
from pi0.policy import Pi0Policy
from pi0.training import trainable_state_dict


class FakePrefixEncoder(nn.Module):
    def __init__(self, source_width: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=source_width))
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)

    def encode_images(self, pixel_values: Tensor) -> Tensor:
        return torch.zeros(pixel_values.shape[0], 2, 16)

    def embed_text(self, input_ids: Tensor) -> Tensor:
        return torch.zeros(input_ids.shape[0], input_ids.shape[1], 16)


def make_normalizer() -> Pi0Normalizer:
    state_stats = NormStats(
        mean=torch.tensor([1.0, 2.0]),
        std=torch.tensor([3.0, 4.0]),
        q01=torch.tensor([-1.0, -2.0]),
        q99=torch.tensor([5.0, 6.0]),
    )
    action_stats = NormStats(
        mean=torch.tensor([7.0, 8.0]),
        std=torch.tensor([9.0, 10.0]),
        q01=torch.tensor([-3.0, -4.0]),
        q99=torch.tensor([11.0, 12.0]),
    )
    return Pi0Normalizer(state_stats, action_stats, use_quantiles=True)


def write_artifact(root: Path, policy: Pi0Policy, *, step: int = 123) -> None:
    root.mkdir()
    save_file(trainable_state_dict(policy), root / "model.safetensors")
    assert policy.normalizer is not None
    save_file(
        {key: value.cpu().contiguous() for key, value in policy.normalizer.state_dict().items()},
        root / "normalizer.safetensors",
    )
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "step": step,
                "training_config": {"seed": 42},
                "model_config": asdict(policy.config),
            }
        ),
        encoding="utf-8",
    )


def test_deploy_artifact_restores_model_and_normalizer(tmp_path: Path) -> None:
    source = Pi0Policy(TINY_PI0, FakePrefixEncoder(), make_normalizer())
    artifact = tmp_path / "artifact"
    write_artifact(artifact, source)
    expected_weight = source.core.action_embedding.action_in_proj.weight.detach().clone()

    target = Pi0Policy(
        TINY_PI0,
        FakePrefixEncoder(),
        load_deploy_normalizer(artifact),
        trainable_dtype=torch.bfloat16,
    )

    with torch.no_grad():
        target.core.action_embedding.action_in_proj.weight.fill_(99.0)

    metadata = load_deploy_weights(
        target,
        artifact,
        device=torch.device("cpu"),
    )

    assert metadata.step == 123
    assert metadata.model_config == TINY_PI0
    assert target.core.action_embedding.action_in_proj.weight.dtype == torch.bfloat16
    assert torch.equal(
        target.core.action_embedding.action_in_proj.weight,
        expected_weight.to(torch.bfloat16),
    )
    assert target.normalizer is not None
    assert torch.equal(target.normalizer.action_q99, torch.tensor([11.0, 12.0]))
    assert not target.training


def test_deploy_artifact_rejects_missing_weight(tmp_path: Path) -> None:
    policy = Pi0Policy(TINY_PI0, FakePrefixEncoder(), make_normalizer())
    artifact = tmp_path / "artifact"
    write_artifact(artifact, policy)
    weights = trainable_state_dict(policy)
    weights.pop(next(iter(weights)))
    save_file(weights, artifact / "model.safetensors")

    with pytest.raises(ValueError, match="Artifact parameter mismatch"):
        load_deploy_weights(
            policy,
            artifact,
            device=torch.device("cpu"),
        )


def test_deploy_metadata_rejects_config_drift(tmp_path: Path) -> None:
    policy = Pi0Policy(TINY_PI0, FakePrefixEncoder(), make_normalizer())
    artifact = tmp_path / "artifact"
    write_artifact(artifact, policy)
    metadata_path = artifact / "metadata.json"
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw["model_config"]["unexpected"] = True
    metadata_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid model_config keys"):
        load_deploy_metadata(artifact)
