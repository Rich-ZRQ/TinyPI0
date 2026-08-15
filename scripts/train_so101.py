"""Train Tiny pi0 on the recorded two-camera SO101 dataset."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from torch.utils.data import DataLoader

from configs import (
    SO101_LARGE,
    SO101_RECOMMENDED,
    TINY_PI0,
    Pi0Config,
    TrainingConfig,
)
from pi0.lerobot_dataset import (
    LeRobotPi0Dataset,
    Pi0DataCollator,
    load_lerobot_normalizer,
)
from pi0.paligemma_prefix import PaliGemmaPrefixEncoder
from pi0.policy import Pi0Policy
from pi0.processor import Pi0Processor
from pi0.training import Pi0Trainer, split_episode_ids


@dataclass(frozen=True)
class Args:
    dataset_root: Path = Path.home() / ".cache/huggingface/lerobot/Rich-RZ/so101_chocolates_to_bowl_v1"
    paligemma_snapshot: Path | None = None
    output_dir: Path = Path("checkpoints/so101_recommended")
    profile: Literal["tiny", "recommended", "large"] = "recommended"
    max_steps: int = 30_000
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_workers: int = 4
    validation_fraction: float = 0.1
    validation_interval: int = 500
    validation_batches: int = 32
    checkpoint_interval: int = 1_000
    gradient_checkpointing: bool = False
    compile_model: bool = False
    resume: bool = False
    seed: int = 42


def find_paligemma_snapshot(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        if not explicit_path.is_dir():
            raise FileNotFoundError(explicit_path)
        return explicit_path

    root = Path.home() / ".cache/huggingface/hub/models--google--paligemma2-3b-pt-224/snapshots"
    snapshots = sorted(root.glob("*"))

    if not snapshots:
        raise FileNotFoundError(f"No PaliGemma snapshot found under {root}; pass --paligemma-snapshot")

    return snapshots[-1]


def select_model_config(profile: str) -> Pi0Config:
    if profile == "tiny":
        return TINY_PI0
    if profile == "recommended":
        return SO101_RECOMMENDED
    if profile == "large":
        return SO101_LARGE
    raise ValueError(f"Unknown model profile: {profile}")


def main(args: Args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("SO101 training requires a CUDA GPU")

    # Model initialization must be seeded before constructing the trainable core.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_config = select_model_config(args.profile)
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        validation_fraction=args.validation_fraction,
        validation_interval=args.validation_interval,
        validation_batches=args.validation_batches,
        checkpoint_interval=args.checkpoint_interval,
        gradient_checkpointing=args.gradient_checkpointing,
        compile_model=args.compile_model,
        seed=args.seed,
    )
    snapshot = find_paligemma_snapshot(args.paligemma_snapshot)

    complete_dataset = LeRobotPi0Dataset(
        root=args.dataset_root,
        config=model_config,
    )
    train_episodes, validation_episodes = split_episode_ids(
        complete_dataset.episode_ids,
        validation_fraction=training_config.validation_fraction,
        seed=training_config.seed,
    )
    train_dataset = LeRobotPi0Dataset(
        root=args.dataset_root,
        config=model_config,
        episodes=train_episodes,
    )
    validation_dataset = LeRobotPi0Dataset(
        root=args.dataset_root,
        config=model_config,
        episodes=validation_episodes,
    )

    processor = Pi0Processor(
        config=model_config,
        snapshot_path=snapshot,
    )
    collator = Pi0DataCollator(processor)
    loader_options = {
        "batch_size": training_config.micro_batch_size,
        "num_workers": training_config.num_workers,
        "collate_fn": collator,
        "pin_memory": True,
        "persistent_workers": training_config.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(training_config.seed),
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )

    prefix_encoder = PaliGemmaPrefixEncoder(
        snapshot_path=snapshot,
        device=device,
        dtype=dtype,
    )
    normalizer = load_lerobot_normalizer(args.dataset_root)
    policy = Pi0Policy(
        config=model_config,
        prefix_encoder=prefix_encoder,
        normalizer=normalizer,
    )
    trainer = Pi0Trainer(
        policy=policy,
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=training_config,
    )

    trainable_parameters = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)
    print("Device:", device)
    print("Frozen frontend dtype:", dtype)
    print("Trainable dtype:", policy.model_dtype)
    print("Profile:", args.profile)
    print("Train/validation episodes:", len(train_episodes), len(validation_episodes))
    print("Train/validation frames:", len(train_dataset), len(validation_dataset))
    print("Micro/effective batch:", training_config.micro_batch_size, training_config.effective_batch_size)
    print("Trainable parameters:", trainable_parameters)

    trainer.fit(resume=args.resume)


if __name__ == "__main__":
    main(tyro.cli(Args))
