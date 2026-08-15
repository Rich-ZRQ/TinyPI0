"""Run one real end-to-end SO-ARM101 pi0 inference smoke test."""

import time
from pathlib import Path

import torch
from PIL import Image

from configs import SO101_RECOMMENDED
from pi0.paligemma_prefix import PaliGemmaPrefixEncoder
from pi0.policy import Pi0Policy
from pi0.processor import Pi0Processor
from pi0.types import IMAGE_KEYS

MODEL_CONFIG = SO101_RECOMMENDED


def find_snapshot() -> Path:
    snapshot_root = Path.home() / ".cache/huggingface/hub/models--google--paligemma2-3b-pt-224/snapshots"

    snapshots = list(snapshot_root.glob("*"))

    if not snapshots:
        raise FileNotFoundError(f"No PaliGemma snapshot found in {snapshot_root}")

    return snapshots[0]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test")

    torch.manual_seed(0)

    snapshot = find_snapshot()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    processor = Pi0Processor(
        config=MODEL_CONFIG,
        snapshot_path=snapshot,
    )

    observation = processor(
        images={
            IMAGE_KEYS[0]: [
                Image.new(
                    "RGB",
                    (640, 480),
                    color=(100, 150, 200),
                )
            ],
            IMAGE_KEYS[1]: [
                Image.new(
                    "RGB",
                    (320, 240),
                    color=(200, 100, 50),
                )
            ],
        },
        prompts=["pick up the object"],
        state=torch.zeros(
            1,
            MODEL_CONFIG.action_dim,
        ),
    )

    prefix_encoder = PaliGemmaPrefixEncoder(
        snapshot_path=snapshot,
        device=device,
        dtype=dtype,
    )

    # 暂时不提供normalizer，因为现在只验证架构，
    # 输出动作不具有真实物理意义。
    policy = Pi0Policy(
        config=MODEL_CONFIG,
        prefix_encoder=prefix_encoder,
        normalizer=None,
    )
    policy.eval()

    trainable_parameters = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)

    frozen_parameters = sum(parameter.numel() for parameter in policy.parameters() if not parameter.requires_grad)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = time.perf_counter()

    actions = policy.sample_actions(
        observation,
        num_steps=2,
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    print("Device:", device)
    print("Dtype:", dtype)
    print(
        "Decoder profile:",
        f"prefix={MODEL_CONFIG.paligemma.width}",
        f"action={MODEL_CONFIG.action_expert.width}",
        f"depth={MODEL_CONFIG.paligemma.depth}",
    )
    print("Actions:", actions.shape)
    print("Actions finite:", torch.isfinite(actions).all().item())
    print("Trainable parameters:", trainable_parameters)
    print("Frozen parameters:", frozen_parameters)
    print(
        "Allocated MiB:",
        torch.cuda.memory_allocated() / 1024**2,
    )
    print(
        "Peak allocated MiB:",
        torch.cuda.max_memory_allocated() / 1024**2,
    )
    print("Elapsed seconds:", elapsed)


if __name__ == "__main__":
    main()
