"""Benchmark scalable Tiny pi0 inference profiles on the local GPU."""

import gc
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from configs.schema import Pi0Config, TransformerConfig, VisionConfig
from pi0.paligemma_prefix import PaliGemmaPrefixEncoder
from pi0.policy import Pi0Policy
from pi0.processor import Pi0Processor
from pi0.types import IMAGE_KEYS


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    prefix_width: int
    action_width: int
    depth: int
    prefix_mlp_dim: int
    action_mlp_dim: int
    num_heads: int
    head_dim: int

    def make_config(self) -> Pi0Config:
        return Pi0Config(
            vision=VisionConfig(
                image_size=224,
                patch_size=14,
                width=128,
                depth=2,
                mlp_dim=256,
                num_heads=4,
                projection_dim=self.prefix_width,
            ),
            paligemma=TransformerConfig(
                width=self.prefix_width,
                depth=self.depth,
                mlp_dim=self.prefix_mlp_dim,
                num_heads=self.num_heads,
                num_kv_heads=1,
                head_dim=self.head_dim,
            ),
            action_expert=TransformerConfig(
                width=self.action_width,
                depth=self.depth,
                mlp_dim=self.action_mlp_dim,
                num_heads=self.num_heads,
                num_kv_heads=1,
                head_dim=self.head_dim,
            ),
            action_dim=32,
            action_horizon=50,
            max_token_len=48,
            dtype="bfloat16",
        )


PROFILES = (
    BenchmarkProfile("128-d2", 128, 64, 2, 256, 128, 4, 32),
    BenchmarkProfile("256-d4", 256, 128, 4, 2048, 512, 4, 64),
    BenchmarkProfile("512-d6", 512, 256, 6, 4096, 1024, 8, 64),
    BenchmarkProfile("768-d8", 768, 384, 8, 6144, 1536, 8, 96),
    BenchmarkProfile("1024-d6", 1024, 512, 6, 8192, 2048, 8, 128),
    BenchmarkProfile("1024-d8", 1024, 512, 8, 8192, 2048, 8, 128),
    BenchmarkProfile("1280-d8", 1280, 640, 8, 10_240, 2560, 10, 128),
    BenchmarkProfile("1536-d8", 1536, 768, 8, 12_288, 3072, 12, 128),
    BenchmarkProfile("1536-d10", 1536, 768, 10, 12_288, 3072, 12, 128),
    BenchmarkProfile("2048-d8", 2048, 1024, 8, 16_384, 4096, 16, 128),
)


def find_snapshot() -> Path:
    snapshot_root = Path.home() / ".cache/huggingface/hub/models--google--paligemma2-3b-pt-224/snapshots"
    snapshots = list(snapshot_root.glob("*"))
    if not snapshots:
        raise FileNotFoundError(f"No PaliGemma snapshot found in {snapshot_root}")
    return snapshots[0]


def parameter_count(policy: Pi0Policy) -> int:
    return sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)


@torch.no_grad()
def sample_from_projected_prefix(
    policy: Pi0Policy,
    prefix_tokens: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    prefix_att_masks: torch.Tensor,
    state: torch.Tensor,
    noise: torch.Tensor,
    *,
    num_steps: int,
) -> torch.Tensor:
    prefix_cache = policy.core.prefill_prefix(
        prefix_tokens,
        prefix_pad_masks,
        prefix_att_masks,
        state,
    )
    actions = noise.clone()
    step_size = -1.0 / num_steps

    for step in range(num_steps):
        timestep = torch.full(
            (state.shape[0],),
            1.0 + step * step_size,
            dtype=torch.float32,
            device=state.device,
        )
        velocity = policy.core.predict_velocity_with_cache(
            prefix_cache,
            prefix_pad_masks,
            state,
            actions,
            timestep,
        )
        actions = actions + step_size * velocity

    return actions


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(0)
    snapshot = find_snapshot()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # The dataset has two 640x480 cameras and a 6-dimensional SO-ARM101 state.
    # State is padded to the model's fixed 32-dimensional external contract.
    processor = Pi0Processor(PROFILES[0].make_config(), snapshot)
    observation = processor(
        images={
            IMAGE_KEYS[0]: [Image.new("RGB", (640, 480), (100, 150, 200))],
            IMAGE_KEYS[1]: [Image.new("RGB", (640, 480), (200, 100, 50))],
        },
        prompts=["put the chocolates into the bowl"],
        state=torch.zeros(1, 32),
    )

    prefix_encoder = PaliGemmaPrefixEncoder(snapshot, device, dtype)
    fixed_noise = torch.zeros(
        1,
        PROFILES[0].make_config().action_horizon,
        PROFILES[0].make_config().action_dim,
        device=device,
        dtype=dtype,
    )
    model_observation = observation.to(device=device, dtype=dtype)

    # Compute the fixed PaliGemma frontend once. Every profile consumes the
    # same 2304-wide source tokens, so only the scalable trainable model needs
    # to be repeated during the profile sweep.
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    frontend_start = time.perf_counter()
    source_parts = [
        prefix_encoder.encode_images(model_observation.images[IMAGE_KEYS[0]]),
        prefix_encoder.encode_images(model_observation.images[IMAGE_KEYS[1]]),
        prefix_encoder.embed_text(model_observation.tokenized_prompt),
    ]
    torch.cuda.synchronize()
    frontend_latency = time.perf_counter() - frontend_start
    frontend_peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    frontend_allocated_mib = torch.cuda.memory_allocated() / 1024**2

    source_tokens = torch.cat(source_parts, dim=1)
    image_mask = torch.ones(
        1,
        source_parts[0].shape[1] + source_parts[1].shape[1],
        dtype=torch.bool,
        device=device,
    )
    prefix_pad_masks = torch.cat(
        [image_mask, model_observation.tokenized_prompt_mask],
        dim=1,
    )
    prefix_att_masks = torch.zeros_like(prefix_pad_masks)
    state = model_observation.state

    frontend_transient_mib = frontend_peak_mib - frontend_allocated_mib
    print(f"Frozen frontend allocated: {frontend_allocated_mib:.1f} MiB")
    print(f"Frozen frontend latency: {frontend_latency:.3f} s")
    print("profile,params_m,core_10step_s,estimated_total_s,estimated_peak_mib,status")

    for profile in PROFILES:
        policy: Pi0Policy | None = None
        try:
            config = profile.make_config()
            policy = Pi0Policy(config, prefix_encoder).eval()
            params_m = parameter_count(policy) / 1e6

            # The trainable model is randomly initialized. Zeroing it keeps
            # every matrix multiplication and cache shape intact while
            # preventing ten Euler steps from diverging before training.
            with torch.no_grad():
                for parameter in policy.core.parameters():
                    parameter.zero_()

            with torch.no_grad():
                prefix_tokens = policy.prefix_embedding.input_projection(source_tokens)

            # Warm up only the profile-dependent kernels. The frozen frontend
            # was measured separately above.
            sample_from_projected_prefix(
                policy,
                prefix_tokens,
                prefix_pad_masks,
                prefix_att_masks,
                state,
                fixed_noise,
                num_steps=2,
            )
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()
            actions = sample_from_projected_prefix(
                policy,
                prefix_tokens,
                prefix_pad_masks,
                prefix_att_masks,
                state,
                fixed_noise,
                num_steps=10,
            )
            torch.cuda.synchronize()
            core_latency = time.perf_counter() - start
            core_peak_mib = torch.cuda.max_memory_allocated() / 1024**2
            model_allocated_mib = torch.cuda.memory_allocated() / 1024**2
            estimated_peak_mib = max(
                core_peak_mib,
                model_allocated_mib + frontend_transient_mib,
            )

            if not torch.isfinite(actions).all():
                raise RuntimeError("non-finite actions")

            print(
                f"{profile.name},{params_m:.1f},{core_latency:.3f},"
                f"{frontend_latency + core_latency:.3f},{estimated_peak_mib:.1f},ok"
            )
        except torch.OutOfMemoryError:
            print(f"{profile.name},NA,NA,NA,oom")
        finally:
            if policy is not None:
                del policy
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
