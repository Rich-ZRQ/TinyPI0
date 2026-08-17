"""Inspect one real SO101 LeRobot sample through the Tiny pi0 adapter."""

from pathlib import Path

from configs import SO101_TINY
from pi0.lerobot_dataset import LeRobotPi0Dataset, load_lerobot_normalizer

DATASET_ROOT = Path.home() / ".cache/huggingface/lerobot/Rich-RZ/so101_chocolates_to_bowl_v1"


def main() -> None:
    dataset = LeRobotPi0Dataset(
        root=DATASET_ROOT,
        config=SO101_TINY,
    )
    sample = dataset[0]
    normalizer = load_lerobot_normalizer(DATASET_ROOT)
    normalized_state = normalizer.normalize_state(sample.state)

    print("Frames:", len(dataset))
    print("Dataset FPS:", dataset.fps)
    print("Robot action/state dims:", dataset.robot_action_dim, dataset.robot_state_dim)
    print("Episode/frame:", sample.episode_index, sample.frame_index)
    print("Timestamp:", sample.timestamp)
    print("Prompt:", sample.prompt)
    print("State shape:", tuple(sample.state.shape))
    print("Normalized robot state:", normalized_state[: dataset.robot_state_dim].tolist())
    print("Action chunk shape:", tuple(sample.actions.shape))
    print("Valid action steps:", int(sample.action_valid_mask.sum()))

    for key, image in sample.images.items():
        print(f"{key}:", None if image is None else image.size)


if __name__ == "__main__":
    main()
