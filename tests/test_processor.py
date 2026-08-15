from pathlib import Path

import torch
from PIL import Image

from configs.tiny import TINY_PI0
from pi0.processor import Pi0Processor
from pi0.types import IMAGE_KEYS


def find_snapshot() -> Path:
    snapshots = Path.home() / ".cache/huggingface/hub/models--google--paligemma2-3b-pt-224/snapshots"

    return next(snapshots.glob("*"))


def test_processor_builds_two_camera_observation() -> None:
    processor = Pi0Processor(
        config=TINY_PI0,
        snapshot_path=find_snapshot(),
    )

    batch_size = 2

    base_images = [
        Image.new(
            "RGB",
            (640, 480),
            color=(100, 150, 200),
        )
        for _ in range(batch_size)
    ]

    wrist_images = [
        Image.new(
            "RGB",
            (320, 240),
            color=(200, 100, 50),
        )
        for _ in range(batch_size)
    ]

    observation = processor(
        images={
            IMAGE_KEYS[0]: base_images,
            IMAGE_KEYS[1]: wrist_images,
        },
        prompts=[
            "pick_up the object",
            "move left",
        ],
        state=torch.zeros(
            batch_size,
            TINY_PI0.action_dim,
        ),
    )

    for key in IMAGE_KEYS:
        assert observation.images[key].shape == (
            batch_size,
            3,
            224,
            224,
        )

    assert torch.all(observation.image_masks[IMAGE_KEYS[0]])
    assert torch.all(observation.image_masks[IMAGE_KEYS[1]])
    assert not torch.any(observation.image_masks[IMAGE_KEYS[2]])

    assert observation.tokenized_prompt.shape == (
        batch_size,
        TINY_PI0.max_token_len,
    )
    assert observation.tokenized_prompt_mask.shape == (
        batch_size,
        TINY_PI0.max_token_len,
    )


def test_tokenizer_uses_bos_and_no_image_placeholders() -> None:
    processor = Pi0Processor(
        config=TINY_PI0,
        snapshot_path=find_snapshot(),
    )

    token_ids, token_mask = processor.tokenize_prompts(["pick up the object"])

    valid_ids = token_ids[0][token_mask[0]]

    assert valid_ids[0].item() == (processor.tokenizer.bos_token_id)

    newline_count = len(processor.newline_token_ids)

    assert valid_ids[-newline_count:].tolist() == (processor.newline_token_ids)

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

    assert image_token_id not in valid_ids.tolist()


def test_processor_supports_missing_sample_image() -> None:
    processor = Pi0Processor(
        config=TINY_PI0,
        snapshot_path=find_snapshot(),
    )

    valid_image = Image.new(
        "RGB",
        (224, 224),
    )

    observation = processor(
        images={
            IMAGE_KEYS[0]: [
                valid_image,
                None,
            ],
        },
        prompts=[
            "task one",
            "task two",
        ],
        state=torch.zeros(
            2,
            TINY_PI0.action_dim,
        ),
    )

    assert observation.image_masks[IMAGE_KEYS[0]].tolist() == [True, False]
