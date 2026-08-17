import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from configs import TINY_PI0, TrainingConfig
from pi0.lerobot_dataset import Pi0TrainingBatch
from pi0.policy import Pi0Policy
from pi0.training import (
    Pi0Trainer,
    cosine_learning_rate,
    create_optimizer,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
    split_episode_ids,
)
from pi0.types import IMAGE_KEYS, Observation


class FakePrefixEncoder(nn.Module):
    def __init__(self, source_width: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=source_width))
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.source_width = source_width

    def encode_images(self, pixel_values: Tensor) -> Tensor:
        values = pixel_values.mean(dim=(1, 2, 3))
        return values[:, None, None].expand(-1, 2, self.source_width)

    def embed_text(self, input_ids: Tensor) -> Tensor:
        return input_ids.to(torch.float32)[:, :, None].expand(
            -1,
            -1,
            self.source_width,
        )


def make_batch() -> Pi0TrainingBatch:
    batch_size = 1
    observation = Observation(
        images={key: torch.ones(batch_size, 3, 224, 224) for key in IMAGE_KEYS},
        image_masks={
            IMAGE_KEYS[0]: torch.ones(batch_size, dtype=torch.bool),
            IMAGE_KEYS[1]: torch.zeros(batch_size, dtype=torch.bool),
            IMAGE_KEYS[2]: torch.zeros(batch_size, dtype=torch.bool),
        },
        state=torch.zeros(batch_size, TINY_PI0.action_dim),
        tokenized_prompt=torch.ones(
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
    return Pi0TrainingBatch(
        observation=observation,
        actions=torch.randn(
            batch_size,
            TINY_PI0.action_horizon,
            TINY_PI0.action_dim,
        ),
        action_valid_mask=torch.cat(
            (
                torch.ones(batch_size, 40, dtype=torch.bool),
                torch.zeros(batch_size, 10, dtype=torch.bool),
            ),
            dim=1,
        ),
        action_dim_mask=torch.cat(
            (
                torch.ones(batch_size, 6, dtype=torch.bool),
                torch.zeros(batch_size, TINY_PI0.action_dim - 6, dtype=torch.bool),
            ),
            dim=1,
        ),
    )


def test_episode_split_is_disjoint_and_deterministic() -> None:
    first = split_episode_ids(range(10), validation_fraction=0.2, seed=7)
    second = split_episode_ids(range(10), validation_fraction=0.2, seed=7)
    training_ids, validation_ids = first

    assert first == second
    assert len(training_ids) == 8
    assert len(validation_ids) == 2
    assert set(training_ids).isdisjoint(validation_ids)
    assert set(training_ids + validation_ids) == set(range(10))


def test_cosine_schedule_warms_up_and_decays(tmp_path: Path) -> None:
    config = TrainingConfig(
        output_dir=tmp_path,
        warmup_steps=10,
        decay_steps=100,
    )

    assert cosine_learning_rate(0, config) == pytest.approx(config.learning_rate / 11)
    assert cosine_learning_rate(10, config) == pytest.approx(config.learning_rate)
    assert cosine_learning_rate(100, config) == pytest.approx(config.end_learning_rate)


def test_checkpoint_round_trip_only_restores_trainable_parameters(
    tmp_path: Path,
) -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    model[0].requires_grad_(False)
    config = TrainingConfig(output_dir=tmp_path)
    optimizer = create_optimizer(model, config)
    original_frozen = model[0].weight.detach().clone()
    original_trainable = model[1].weight.detach().clone()

    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    checkpoint_path = save_checkpoint(
        model=model,
        optimizer=optimizer,
        config=config,
        step=12,
    )
    saved_trainable = model[1].weight.detach().clone()

    with torch.no_grad():
        model[0].weight.add_(10)
        model[1].weight.add_(10)

    restored_step = load_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
    )

    assert restored_step == 12
    assert torch.equal(model[0].weight, original_frozen + 10)
    assert torch.equal(model[1].weight, saved_trainable)
    assert not torch.equal(model[1].weight, original_trainable)
    assert latest_checkpoint(tmp_path) == checkpoint_path


def test_optimizer_rejects_bfloat16_master_parameters(tmp_path: Path) -> None:
    model = nn.Linear(2, 2).to(torch.bfloat16)

    with pytest.raises(TypeError, match="must remain FP32"):
        create_optimizer(model, TrainingConfig(output_dir=tmp_path))


def test_resume_rejects_legacy_bfloat16_master_checkpoint(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    config = TrainingConfig(output_dir=tmp_path)
    optimizer = create_optimizer(model, config)
    checkpoint = save_checkpoint(
        model=model,
        optimizer=optimizer,
        config=config,
        step=1,
    )
    weights_path = checkpoint / "model.safetensors"
    legacy_weights = {name: value.to(torch.bfloat16) for name, value in load_file(weights_path).items()}
    save_file(legacy_weights, weights_path)

    with pytest.raises(TypeError, match="legacy BF16-master"):
        load_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_path=checkpoint,
            device=torch.device("cpu"),
        )


def test_small_learning_rate_updates_fp32_master_parameter(tmp_path: Path) -> None:
    parameter = nn.Parameter(torch.tensor([0.02], dtype=torch.float32))
    model = nn.ParameterList([parameter])
    optimizer = create_optimizer(
        model,
        TrainingConfig(
            output_dir=tmp_path,
            learning_rate=2.5e-5,
            end_learning_rate=2.5e-6,
        ),
    )
    before = parameter.detach().clone()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    assert not torch.equal(parameter, before)


def test_trainer_runs_accumulation_validation_and_checkpoint(tmp_path: Path) -> None:
    policy = Pi0Policy(TINY_PI0, FakePrefixEncoder())
    batch = make_batch()
    config = TrainingConfig(
        output_dir=tmp_path,
        max_steps=1,
        micro_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=1,
        decay_steps=2,
        validation_interval=1,
        validation_batches=1,
        checkpoint_interval=1,
        num_workers=0,
        gradient_checkpointing=True,
    )
    trainer = Pi0Trainer(
        policy=policy,
        train_loader=[batch, batch],
        validation_loader=[batch],
        config=config,
    )

    metrics = trainer.fit()

    assert len(metrics) == 1
    assert metrics[0].step == 1
    assert metrics[0].validation_loss is not None
    assert metrics[0].validation_action_mae is not None
    assert metrics[0].validation_first_action_mae is not None
    assert torch.isfinite(torch.tensor(metrics[0].train_loss))
    checkpoint = latest_checkpoint(tmp_path)
    assert checkpoint is not None
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["trainable_dtype"] == "float32"
    assert metadata["metrics"]["validation_action_mae"] is not None
    metric_lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(metric_lines) == 1
    assert json.loads(metric_lines[0])["step"] == 1
