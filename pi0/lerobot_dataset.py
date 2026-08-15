"""LeRobot v3 dataset adapter for Tiny pi0 training."""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, TypeAlias

import av
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from configs.schema import Pi0Config
from pi0.normalization import NormStats, Pi0Normalizer
from pi0.processor import Pi0Processor
from pi0.types import IMAGE_KEYS, Observation

VideoDecoder: TypeAlias = Callable[[Path, float], Image.Image]


@dataclass(frozen=True)
class Pi0TrainingSample:
    """One raw observation and its future action chunk."""

    images: dict[str, Image.Image | None]
    prompt: str
    state: Tensor
    actions: Tensor
    action_valid_mask: Tensor
    episode_index: int
    frame_index: int
    timestamp: float


@dataclass(frozen=True)
class Pi0TrainingBatch:
    """A processor-ready batch consumed by ``Pi0Policy.compute_loss``."""

    observation: Observation
    actions: Tensor
    action_valid_mask: Tensor

    def masked_mean_loss(self, per_step_loss: Tensor) -> Tensor:
        """Average a [B, H] loss over non-padding action steps."""

        if per_step_loss.shape != self.action_valid_mask.shape:
            raise ValueError(
                "per_step_loss and action_valid_mask must have the same shape, "
                f"got {tuple(per_step_loss.shape)} and {tuple(self.action_valid_mask.shape)}"
            )

        mask = self.action_valid_mask.to(
            device=per_step_loss.device,
            dtype=per_step_loss.dtype,
        )
        return (per_step_loss * mask).sum() / mask.sum().clamp_min(1)


def make_action_chunk(
    actions: Tensor,
    *,
    start: int,
    episode_end: int,
    horizon: int,
) -> tuple[Tensor, Tensor]:
    """Slice future actions without crossing an episode boundary.

    Missing tail steps repeat the episode's final action. The returned Boolean
    mask distinguishes real targets from repeated padding.
    """

    if actions.ndim != 2:
        raise ValueError(f"actions must have shape [N, D], got {tuple(actions.shape)}")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if not 0 <= start < episode_end <= actions.shape[0]:
        raise ValueError(
            "expected 0 <= start < episode_end <= number of actions, "
            f"got start={start}, episode_end={episode_end}, N={actions.shape[0]}"
        )

    valid_length = min(horizon, episode_end - start)
    chunk = actions[start : start + valid_length]

    if valid_length < horizon:
        padding = actions[episode_end - 1].expand(horizon - valid_length, -1)
        chunk = torch.cat((chunk, padding), dim=0)

    valid_mask = torch.arange(horizon) < valid_length
    return chunk, valid_mask


def decode_video_frame(path: Path, timestamp: float) -> Image.Image:
    """Decode the video frame nearest to ``timestamp`` using PyAV."""

    if timestamp < 0:
        raise ValueError(f"timestamp must be non-negative, got {timestamp}")

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = float(stream.time_base)
        seek_timestamp = max(timestamp - 1.0, 0.0)
        container.seek(
            int(seek_timestamp / time_base),
            stream=stream,
            backward=True,
        )

        best_frame = None
        best_distance = float("inf")

        for frame in container.decode(stream):
            if frame.time is None:
                continue

            distance = abs(float(frame.time) - timestamp)

            if distance < best_distance:
                best_frame = frame
                best_distance = distance

            if float(frame.time) > timestamp and best_frame is not None:
                break

        if best_frame is None:
            raise RuntimeError(f"Could not decode a frame from {path} at {timestamp:.6f}s")

        return best_frame.to_image().convert("RGB")


class LeRobotPi0Dataset(Dataset[Pi0TrainingSample]):
    """Read a local LeRobot v3 dataset with pi0 tensor contracts."""

    CAMERA_MAP: ClassVar[dict[str, str]] = {
        "observation.images.front": IMAGE_KEYS[0],
        "observation.images.wrist": IMAGE_KEYS[1],
    }

    def __init__(
        self,
        *,
        root: str | Path,
        config: Pi0Config,
        episodes: Sequence[int] | None = None,
        video_decoder: VideoDecoder = decode_video_frame,
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.config = config
        self.video_decoder = video_decoder
        self.info = self._read_json(self.root / "meta/info.json")
        self.fps = int(self.info["fps"])

        if self.fps <= 0:
            raise ValueError(f"Dataset fps must be positive, got {self.fps}")

        self._validate_features()
        self._load_frames()
        self._load_episode_metadata()
        self._load_tasks()

        selected_episodes = set(self._episodes) if episodes is None else set(episodes)
        unknown_episodes = selected_episodes - set(self._episodes)

        if unknown_episodes:
            raise ValueError(f"Unknown episode indices: {sorted(unknown_episodes)}")

        self.sample_indices = [
            index
            for index, episode_index in enumerate(self.episode_indices.tolist())
            if episode_index in selected_episodes
        ]

    def __len__(self) -> int:
        return len(self.sample_indices)

    @property
    def episode_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._episodes))

    def __getitem__(self, item: int) -> Pi0TrainingSample:
        global_index = self.sample_indices[item]
        episode_index = int(self.episode_indices[global_index])
        episode = self._episodes[episode_index]

        actions, action_valid_mask = make_action_chunk(
            self.actions,
            start=global_index,
            episode_end=episode["dataset_to_index"],
            horizon=self.config.action_horizon,
        )

        timestamp = float(self.timestamps[global_index])
        images: dict[str, Image.Image | None] = {key: None for key in IMAGE_KEYS}

        for source_key, model_key in self.CAMERA_MAP.items():
            video = episode["videos"][source_key]
            video_path = self._video_path(
                source_key,
                video["chunk_index"],
                video["file_index"],
            )
            images[model_key] = self.video_decoder(
                video_path,
                timestamp + video["from_timestamp"],
            )

        return Pi0TrainingSample(
            images=images,
            prompt=self.tasks[int(self.task_indices[global_index])],
            state=self.states[global_index],
            actions=actions,
            action_valid_mask=action_valid_mask,
            episode_index=episode_index,
            frame_index=int(self.frame_indices[global_index]),
            timestamp=timestamp,
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(path)
        return json.loads(path.read_text())

    def _validate_features(self) -> None:
        features = self.info["features"]

        for key in ("action", "observation.state", *self.CAMERA_MAP):
            if key not in features:
                raise ValueError(f"Dataset is missing required feature {key!r}")

        action_dim = int(features["action"]["shape"][0])
        state_dim = int(features["observation.state"]["shape"][0])

        if action_dim > self.config.action_dim or state_dim > self.config.action_dim:
            raise ValueError(
                f"Robot dimensions action={action_dim}, state={state_dim} exceed model action_dim={self.config.action_dim}"
            )

        self.robot_action_dim = action_dim
        self.robot_state_dim = state_dim

    def _load_frames(self) -> None:
        paths = sorted((self.root / "data").glob("chunk-*/*.parquet"))

        if not paths:
            raise FileNotFoundError(f"No frame parquet files found under {self.root / 'data'}")

        columns = [
            "action",
            "observation.state",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ]
        table = pa.concat_tables([pq.read_table(path, columns=columns) for path in paths]).sort_by(
            [("index", "ascending")]
        )

        self.actions = self._pad_vectors(table["action"].to_pylist(), self.robot_action_dim)
        self.states = self._pad_vectors(table["observation.state"].to_pylist(), self.robot_state_dim)
        self.timestamps = torch.tensor(table["timestamp"].to_pylist(), dtype=torch.float64)
        self.frame_indices = torch.tensor(table["frame_index"].to_pylist(), dtype=torch.long)
        self.episode_indices = torch.tensor(table["episode_index"].to_pylist(), dtype=torch.long)
        self.task_indices = torch.tensor(table["task_index"].to_pylist(), dtype=torch.long)

    def _pad_vectors(self, values: list[list[float]], robot_dim: int) -> Tensor:
        tensor = torch.tensor(values, dtype=torch.float32)
        padded = torch.zeros(
            tensor.shape[0],
            self.config.action_dim,
            dtype=torch.float32,
        )
        padded[:, :robot_dim] = tensor
        return padded

    def _load_episode_metadata(self) -> None:
        paths = sorted((self.root / "meta/episodes").glob("chunk-*/*.parquet"))

        if not paths:
            raise FileNotFoundError(f"No episode parquet files found under {self.root / 'meta/episodes'}")

        columns = [
            "episode_index",
            "dataset_from_index",
            "dataset_to_index",
        ]

        for camera_key in self.CAMERA_MAP:
            video_key = f"videos/{camera_key}"
            columns.extend(
                (
                    f"{video_key}/chunk_index",
                    f"{video_key}/file_index",
                    f"{video_key}/from_timestamp",
                )
            )

        table = pa.concat_tables([pq.read_table(path, columns=columns) for path in paths])
        self._episodes = {}

        for row in table.to_pylist():
            episode_index = int(row["episode_index"])
            self._episodes[episode_index] = {
                "dataset_from_index": int(row["dataset_from_index"]),
                "dataset_to_index": int(row["dataset_to_index"]),
                "videos": {
                    camera_key: {
                        "chunk_index": int(row[f"videos/{camera_key}/chunk_index"]),
                        "file_index": int(row[f"videos/{camera_key}/file_index"]),
                        "from_timestamp": float(row[f"videos/{camera_key}/from_timestamp"]),
                    }
                    for camera_key in self.CAMERA_MAP
                },
            }

    def _load_tasks(self) -> None:
        path = self.root / "meta/tasks.parquet"
        table = pq.read_table(path, columns=["task_index", "task"])
        self.tasks = {int(row["task_index"]): str(row["task"]) for row in table.to_pylist()}

    def _video_path(
        self,
        video_key: str,
        chunk_index: int,
        file_index: int,
    ) -> Path:
        relative_path = self.info["video_path"].format(
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        path = self.root / relative_path

        if not path.is_file():
            raise FileNotFoundError(path)

        return path


class Pi0DataCollator:
    """Convert raw LeRobot samples into a model-ready training batch."""

    def __init__(self, processor: Pi0Processor) -> None:
        self.processor = processor

    def __call__(self, samples: Sequence[Pi0TrainingSample]) -> Pi0TrainingBatch:
        if not samples:
            raise ValueError("Cannot collate an empty batch")

        observation = self.processor(
            images={key: [sample.images[key] for sample in samples] for key in IMAGE_KEYS},
            prompts=[sample.prompt for sample in samples],
            state=torch.stack([sample.state for sample in samples]),
        )

        return Pi0TrainingBatch(
            observation=observation,
            actions=torch.stack([sample.actions for sample in samples]),
            action_valid_mask=torch.stack([sample.action_valid_mask for sample in samples]),
        )


def load_lerobot_normalizer(
    root: str | Path,
    *,
    use_quantiles: bool = True,
) -> Pi0Normalizer:
    """Build a normalizer from LeRobot's recorded state/action statistics."""

    path = Path(root) / "meta/stats.json"

    if not path.is_file():
        raise FileNotFoundError(path)

    raw = json.loads(path.read_text())

    def convert(key: str) -> NormStats:
        if key not in raw:
            raise ValueError(f"Dataset statistics are missing {key!r}")

        values = raw[key]
        return NormStats(
            mean=torch.tensor(values["mean"], dtype=torch.float32),
            std=torch.tensor(values["std"], dtype=torch.float32),
            q01=torch.tensor(values["q01"], dtype=torch.float32),
            q99=torch.tensor(values["q99"], dtype=torch.float32),
        )

    return Pi0Normalizer(
        state_stats=convert("observation.state"),
        action_stats=convert("action"),
        use_quantiles=use_quantiles,
    )
