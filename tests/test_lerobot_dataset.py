import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from configs import TINY_PI0
from pi0.lerobot_dataset import (
    LeRobotPi0Dataset,
    Pi0TrainingBatch,
    load_lerobot_normalizer,
    make_action_chunk,
)


def test_make_action_chunk_stays_inside_episode() -> None:
    actions = torch.arange(18, dtype=torch.float32).reshape(6, 3)

    chunk, valid_mask = make_action_chunk(
        actions,
        start=2,
        episode_end=4,
        horizon=5,
    )

    assert torch.equal(chunk[:2], actions[2:4])
    assert torch.equal(chunk[2:], actions[3].expand(3, -1))
    assert valid_mask.tolist() == [True, True, False, False, False]


def test_training_batch_masks_padded_loss() -> None:
    batch = Pi0TrainingBatch(
        observation=None,  # type: ignore[arg-type]
        actions=torch.empty(1, 3, 2),
        action_valid_mask=torch.tensor([[True, True, False]]),
        action_dim_mask=torch.tensor([[True, True]]),
    )

    loss = batch.masked_mean_loss(torch.tensor([[1.0, 3.0, 100.0]]))

    assert loss.item() == pytest.approx(2.0)


def test_dataset_maps_so101_and_pads_episode_tail(tmp_path: Path) -> None:
    _write_synthetic_dataset(tmp_path)
    decode_calls: list[tuple[Path, float]] = []

    def fake_decoder(path: Path, timestamp: float) -> Image.Image:
        decode_calls.append((path, timestamp))
        return Image.new("RGB", (8, 6))

    dataset = LeRobotPi0Dataset(
        root=tmp_path,
        config=TINY_PI0,
        video_decoder=fake_decoder,
    )

    sample = dataset[1]

    assert len(dataset) == 3
    assert dataset.fps == 20
    assert sample.prompt == "pick up chocolate"
    assert sample.state.shape == (TINY_PI0.action_dim,)
    assert torch.equal(sample.state[:2], torch.tensor([3.0, 4.0]))
    assert torch.count_nonzero(sample.state[2:]) == 0
    assert sample.actions.shape == (TINY_PI0.action_horizon, TINY_PI0.action_dim)
    assert torch.equal(sample.actions[0, :2], torch.tensor([30.0, 40.0]))
    assert torch.equal(sample.actions[-1, :2], torch.tensor([30.0, 40.0]))
    assert sample.action_valid_mask.sum().item() == 1
    assert sample.action_dim_mask[:2].tolist() == [True, True]
    assert not torch.any(sample.action_dim_mask[2:])
    assert sample.images["base_0_rgb"] is not None
    assert sample.images["left_wrist_0_rgb"] is not None
    assert sample.images["right_wrist_0_rgb"] is None
    assert len(decode_calls) == 2
    assert decode_calls[0][1] == pytest.approx(10.05)


def test_load_lerobot_normalizer_only_transforms_robot_dims(tmp_path: Path) -> None:
    _write_synthetic_dataset(tmp_path)
    normalizer = load_lerobot_normalizer(tmp_path, use_quantiles=True)
    values = torch.tensor([[0.0, 20.0, 7.0]])

    normalized = normalizer.normalize_state(values)

    assert torch.allclose(normalized, torch.tensor([[-1.0, 1.0, 7.0]]))
    assert torch.allclose(normalizer.unnormalize_actions(normalized), values)


def _write_synthetic_dataset(root: Path) -> None:
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)

    info = {
        "fps": 20,
        "features": {
            "action": {"shape": [2]},
            "observation.state": {"shape": [2]},
            "observation.images.front": {"shape": [6, 8, 3]},
            "observation.images.wrist": {"shape": [6, 8, 3]},
        },
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (root / "meta/info.json").write_text(json.dumps(info))

    frames = pa.table(
        {
            "action": [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
            "observation.state": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "timestamp": [0.0, 0.05, 0.0],
            "frame_index": [0, 1, 0],
            "episode_index": [0, 0, 1],
            "index": [0, 1, 2],
            "task_index": [0, 0, 0],
        }
    )
    pq.write_table(frames, root / "data/chunk-000/file-000.parquet")

    episode_columns = {
        "episode_index": [0, 1],
        "dataset_from_index": [0, 2],
        "dataset_to_index": [2, 3],
    }

    for camera in ("observation.images.front", "observation.images.wrist"):
        prefix = f"videos/{camera}"
        episode_columns[f"{prefix}/chunk_index"] = [0, 0]
        episode_columns[f"{prefix}/file_index"] = [0, 0]
        episode_columns[f"{prefix}/from_timestamp"] = [10.0, 20.0]

    episodes = pa.table(episode_columns)
    pq.write_table(episodes, root / "meta/episodes/chunk-000/file-000.parquet")

    tasks = pa.table({"task_index": [0], "task": ["pick up chocolate"]})
    pq.write_table(tasks, root / "meta/tasks.parquet")

    stats = {
        key: {
            "mean": [5.0, 10.0],
            "std": [2.0, 4.0],
            "q01": [0.0, 0.0],
            "q99": [10.0, 20.0],
        }
        for key in ("observation.state", "action")
    }
    (root / "meta/stats.json").write_text(json.dumps(stats))

    for camera in ("observation.images.front", "observation.images.wrist"):
        path = root / f"videos/{camera}/chunk-000/file-000.mp4"
        path.parent.mkdir(parents=True)
        path.touch()
