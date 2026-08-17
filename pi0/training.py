"""Single-GPU pi0 training, validation and checkpoint utilities."""

import json
import math
import random
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from configs.training import TrainingConfig
from pi0.flow_matching import sample_noise, sample_time
from pi0.lerobot_dataset import Pi0TrainingBatch
from pi0.policy import Pi0Policy


@dataclass(frozen=True)
class StepMetrics:
    step: int
    train_loss: float
    learning_rate: float
    grad_norm: float
    validation_loss: float | None = None
    validation_action_mae: float | None = None
    validation_first_action_mae: float | None = None


@dataclass(frozen=True)
class ValidationMetrics:
    loss: float
    action_mae: float
    first_action_mae: float


def split_episode_ids(
    episode_ids: Sequence[int],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Create deterministic disjoint train/validation episode splits."""

    unique_ids = sorted(set(episode_ids))

    if len(unique_ids) < 2:
        raise ValueError("At least two episodes are required for train/validation splitting")
    if not 0 < validation_fraction < 1:
        raise ValueError(f"validation_fraction must be between zero and one, got {validation_fraction}")

    generator = random.Random(seed)
    generator.shuffle(unique_ids)
    validation_count = max(1, round(len(unique_ids) * validation_fraction))
    validation_count = min(validation_count, len(unique_ids) - 1)
    validation_ids = sorted(unique_ids[:validation_count])
    training_ids = sorted(unique_ids[validation_count:])
    return training_ids, validation_ids


def cosine_learning_rate(step: int, config: TrainingConfig) -> float:
    """Official-style linear warmup followed by cosine decay."""

    if step < 0:
        raise ValueError(f"step cannot be negative, got {step}")

    initial_learning_rate = config.learning_rate / (config.warmup_steps + 1)

    if step < config.warmup_steps:
        return initial_learning_rate + ((config.learning_rate - initial_learning_rate) * step / config.warmup_steps)

    progress = min(
        1.0,
        (step - config.warmup_steps) / max(1, config.decay_steps - config.warmup_steps),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.end_learning_rate + (config.learning_rate - config.end_learning_rate) * cosine


def create_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.AdamW:
    """Create AdamW over trainable parameters only."""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    if not parameters:
        raise ValueError("Model has no trainable parameters")

    low_precision = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.dtype != torch.float32
    ]

    if low_precision:
        raise TypeError(
            "Trainable parameters must remain FP32 so AdamW keeps FP32 master weights and moments; "
            f"low-precision parameters include {low_precision[:8]}"
        )

    return torch.optim.AdamW(
        parameters,
        lr=cosine_learning_rate(0, config),
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )


def trainable_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Return only learned tensors; frozen PaliGemma weights are reloaded separately."""

    return {
        name: parameter.detach().to(device="cpu").contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    step: int,
    flow_generator: torch.Generator | None = None,
    metrics: StepMetrics | None = None,
) -> Path:
    """Atomically save trainable weights, optimizer state and run metadata."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"step-{step:08d}"
    temporary_path = output_dir / f".tmp-step-{step:08d}"

    if temporary_path.exists():
        shutil.rmtree(temporary_path)
    temporary_path.mkdir(parents=True)

    learned_state = trainable_state_dict(model)

    if not learned_state:
        raise ValueError("Model has no trainable parameters to checkpoint")

    save_file(learned_state, temporary_path / "model.safetensors")
    torch.save(optimizer.state_dict(), temporary_path / "optimizer.pt")

    normalizer = getattr(model, "normalizer", None)

    if isinstance(normalizer, nn.Module):
        normalizer_state = {
            name: value.detach().to(device="cpu").contiguous() for name, value in normalizer.state_dict().items()
        }

        if normalizer_state:
            save_file(
                normalizer_state,
                temporary_path / "normalizer.safetensors",
            )

    if flow_generator is not None:
        torch.save(
            flow_generator.get_state(),
            temporary_path / "flow_generator_state.pt",
        )

    serialized_config = asdict(config)
    serialized_config["output_dir"] = str(config.output_dir)
    metadata = {
        "step": step,
        "training_config": serialized_config,
        "trainable_dtype": str(next(iter(learned_state.values())).dtype).removeprefix("torch."),
    }

    if metrics is not None:
        metadata["metrics"] = asdict(metrics)

    model_config = getattr(model, "config", None)

    if model_config is not None:
        metadata["model_config"] = asdict(model_config)
    (temporary_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    if final_path.exists():
        shutil.rmtree(final_path)
    temporary_path.rename(final_path)
    return final_path


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the checkpoint with the largest numeric step."""

    root = Path(output_dir)

    if not root.is_dir():
        return None

    candidates: list[tuple[int, Path]] = []

    for path in root.glob("step-*"):
        if not path.is_dir():
            continue

        try:
            step = int(path.name.removeprefix("step-"))
        except ValueError:
            continue

        candidates.append((step, path))

    return None if not candidates else max(candidates, key=lambda item: item[0])[1]


def load_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | Path,
    device: torch.device,
    flow_generator: torch.Generator | None = None,
) -> int:
    """Restore learned weights and optimizer state, returning the completed step."""

    path = Path(checkpoint_path)
    weights_path = path / "model.safetensors"
    optimizer_path = path / "optimizer.pt"
    metadata_path = path / "metadata.json"

    for required_path in (weights_path, optimizer_path, metadata_path):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    weights = load_file(weights_path, device=str(device))
    trainable_parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    trainable_names = set(trainable_parameters)

    if set(weights) != trainable_names:
        missing = sorted(trainable_names - set(weights))
        unexpected = sorted(set(weights) - trainable_names)
        raise ValueError(f"Checkpoint parameter mismatch: missing={missing}, unexpected={unexpected}")

    dtype_mismatches = [
        f"{name}: checkpoint={weights[name].dtype}, model={parameter.dtype}"
        for name, parameter in trainable_parameters.items()
        if weights[name].dtype != parameter.dtype
    ]

    if dtype_mismatches:
        raise TypeError(
            "Training checkpoint dtype mismatch. Do not resume a legacy BF16-master run as FP32 training; "
            f"restart training instead. Mismatches include {dtype_mismatches[:8]}"
        )

    model.load_state_dict(weights, strict=False)

    normalizer_path = path / "normalizer.safetensors"
    normalizer = getattr(model, "normalizer", None)

    if normalizer_path.is_file():
        if not isinstance(normalizer, nn.Module):
            raise ValueError("Checkpoint contains normalization state but model has no normalizer")
        normalizer.load_state_dict(load_file(normalizer_path, device=str(device)), strict=True)

    optimizer_state = torch.load(
        optimizer_path,
        map_location=device,
        weights_only=True,
    )
    optimizer.load_state_dict(optimizer_state)

    generator_state_path = path / "flow_generator_state.pt"

    if flow_generator is not None:
        if not generator_state_path.is_file():
            raise FileNotFoundError(generator_state_path)

        flow_generator.set_state(
            torch.load(
                generator_state_path,
                map_location="cpu",
                weights_only=True,
            )
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return int(metadata["step"])


class Pi0Trainer:
    """Train a policy on one GPU with accumulation and periodic validation."""

    def __init__(
        self,
        *,
        policy: Pi0Policy,
        train_loader: Iterable[Pi0TrainingBatch],
        validation_loader: Iterable[Pi0TrainingBatch],
        config: TrainingConfig,
    ) -> None:
        self.policy = policy
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.config = config
        self.device = policy.model_device
        self.policy.core.transformer.set_gradient_checkpointing(config.gradient_checkpointing)
        self.optimizer = create_optimizer(policy, config)
        self.loss_function = policy.compute_loss
        self.flow_generator = torch.Generator(device=self.device).manual_seed(config.seed)
        self.autocast_enabled = config.bfloat16_autocast and self.device.type == "cuda"

        if config.compile_model:
            self.loss_function = torch.compile(self.loss_function)

    def fit(self, *, resume: bool = False) -> list[StepMetrics]:
        """Run until ``max_steps`` optimizer updates have completed."""

        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)
        start_step = 0

        if resume:
            checkpoint_path = latest_checkpoint(self.config.output_dir)

            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoint found under {self.config.output_dir}")

            start_step = load_checkpoint(
                model=self.policy,
                optimizer=self.optimizer,
                checkpoint_path=checkpoint_path,
                device=self.device,
                flow_generator=self.flow_generator,
            )

        if start_step >= self.config.max_steps:
            return []

        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        metrics: list[StepMetrics] = []
        global_step = start_step
        accumulation_count = 0
        accumulated_loss = 0.0

        while global_step < self.config.max_steps:
            produced_batch = False

            for batch in self.train_loader:
                produced_batch = True
                noise = sample_noise(
                    tuple(batch.actions.shape),
                    device=self.device,
                    dtype=self.policy.model_dtype,
                    generator=self.flow_generator,
                )
                timestep = sample_time(
                    batch.actions.shape[0],
                    device=self.device,
                    generator=self.flow_generator,
                )
                with self._autocast():
                    per_step_loss = self.loss_function(
                        batch.observation,
                        batch.actions,
                        noise=noise,
                        timestep=timestep,
                        action_dim_mask=batch.action_dim_mask,
                    )
                loss = batch.masked_mean_loss(per_step_loss)

                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at step {global_step}: {loss.item()}")

                (loss / self.config.gradient_accumulation_steps).backward()
                accumulated_loss += float(loss.detach())
                accumulation_count += 1
                del loss, per_step_loss, noise, timestep

                if accumulation_count < self.config.gradient_accumulation_steps:
                    continue

                learning_rate = cosine_learning_rate(global_step, self.config)

                for parameter_group in self.optimizer.param_groups:
                    parameter_group["lr"] = learning_rate

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in self.policy.parameters() if parameter.requires_grad),
                    max_norm=self.config.max_grad_norm,
                )

                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"Non-finite gradient norm at step {global_step}: {grad_norm.item()}")

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                non_finite_parameters = [
                    name
                    for name, parameter in self.policy.named_parameters()
                    if parameter.requires_grad and not torch.isfinite(parameter).all()
                ]

                if non_finite_parameters:
                    raise FloatingPointError(f"Optimizer produced non-finite parameters: {non_finite_parameters[:8]}")

                global_step += 1

                validation_metrics = None

                if global_step % self.config.validation_interval == 0:
                    validation_metrics = self.validate()
                    self.policy.train()

                train_loss = accumulated_loss / accumulation_count
                step_metrics = StepMetrics(
                    step=global_step,
                    train_loss=train_loss,
                    learning_rate=learning_rate,
                    grad_norm=float(grad_norm),
                    validation_loss=(None if validation_metrics is None else validation_metrics.loss),
                    validation_action_mae=(None if validation_metrics is None else validation_metrics.action_mae),
                    validation_first_action_mae=(
                        None if validation_metrics is None else validation_metrics.first_action_mae
                    ),
                )
                metrics.append(step_metrics)
                self._print_metrics(step_metrics)
                self._record_metrics(step_metrics, resume=resume)
                accumulation_count = 0
                accumulated_loss = 0.0

                if global_step % self.config.checkpoint_interval == 0:
                    save_checkpoint(
                        model=self.policy,
                        optimizer=self.optimizer,
                        config=self.config,
                        step=global_step,
                        flow_generator=self.flow_generator,
                        metrics=step_metrics,
                    )

                if global_step >= self.config.max_steps:
                    break

            if not produced_batch:
                raise RuntimeError("Training loader produced no batches")

        if global_step % self.config.checkpoint_interval != 0:
            save_checkpoint(
                model=self.policy,
                optimizer=self.optimizer,
                config=self.config,
                step=global_step,
                flow_generator=self.flow_generator,
                metrics=metrics[-1],
            )

        return metrics

    @torch.no_grad()
    def validate(self) -> ValidationMetrics:
        """Compute velocity loss and physical-action metrics on held-out episodes."""

        self.policy.eval()
        losses: list[float] = []
        action_absolute_error = 0.0
        action_element_count = 0
        first_action_absolute_error = 0.0
        first_action_element_count = 0
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed + 1)

        for batch_index, batch in enumerate(self.validation_loader):
            if batch_index >= self.config.validation_batches:
                break

            batch_size = batch.actions.shape[0]
            noise = torch.randn(
                batch.actions.shape,
                device=self.device,
                dtype=self.policy.model_dtype,
                generator=generator,
            )
            timestep = ((torch.arange(batch_size, device=self.device, dtype=torch.float32) + batch_index) % 9 + 1) / 10
            with self._autocast():
                per_step_loss = self.loss_function(
                    batch.observation,
                    batch.actions,
                    noise=noise,
                    timestep=timestep,
                    action_dim_mask=batch.action_dim_mask,
                )
            loss = batch.masked_mean_loss(per_step_loss)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite validation loss: {loss.item()}")

            losses.append(float(loss))

            if batch_index < self.config.validation_action_batches:
                sampling_noise = torch.randn(
                    batch.actions.shape,
                    device=self.device,
                    dtype=self.policy.model_dtype,
                    generator=generator,
                )
                with self._autocast():
                    predicted_actions = self.policy.sample_actions(
                        batch.observation,
                        num_steps=self.config.validation_sampling_steps,
                        noise=sampling_noise,
                        action_dim_mask=batch.action_dim_mask,
                    )
                target_actions = batch.actions.to(
                    device=self.device,
                    dtype=torch.float32,
                )
                time_mask = batch.action_valid_mask.to(device=self.device)[:, :, None]
                dimension_mask = batch.action_dim_mask.to(device=self.device)[:, None, :]
                action_mask = time_mask & dimension_mask
                absolute_error = (predicted_actions - target_actions).abs()
                action_absolute_error += float((absolute_error * action_mask).sum())
                action_element_count += int(action_mask.sum())

                first_mask = batch.action_dim_mask.to(device=self.device)
                first_error = absolute_error[:, 0, :]
                first_action_absolute_error += float((first_error * first_mask).sum())
                first_action_element_count += int(first_mask.sum())

        if not losses:
            raise RuntimeError("Validation loader produced no batches")

        if action_element_count == 0 or first_action_element_count == 0:
            raise RuntimeError("Validation action masks contain no real robot dimensions")

        return ValidationMetrics(
            loss=sum(losses) / len(losses),
            action_mae=action_absolute_error / action_element_count,
            first_action_mae=first_action_absolute_error / first_action_element_count,
        )

    @staticmethod
    def _print_metrics(metrics: StepMetrics) -> None:
        message = (
            f"step={metrics.step} train_loss={metrics.train_loss:.6f} "
            f"lr={metrics.learning_rate:.3e} grad_norm={metrics.grad_norm:.4f}"
        )

        if metrics.validation_loss is not None:
            message += f" validation_loss={metrics.validation_loss:.6f}"

        if metrics.validation_action_mae is not None:
            message += f" validation_action_mae={metrics.validation_action_mae:.4f}"

        if metrics.validation_first_action_mae is not None:
            message += f" validation_first_action_mae={metrics.validation_first_action_mae:.4f}"

        print(message, flush=True)

    def _record_metrics(self, metrics: StepMetrics, *, resume: bool) -> None:
        path = Path(self.config.output_dir) / "metrics.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)

        if metrics.step == 1 and not resume:
            path.write_text("", encoding="utf-8")

        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(metrics)) + "\n")

    def _autocast(self) -> torch.autocast:
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.autocast_enabled,
        )
