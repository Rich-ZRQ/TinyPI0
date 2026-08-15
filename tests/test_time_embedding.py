import pytest
import torch

from pi0.time_embedding import (
    create_sinusoidal_pos_embedding,
)


def test_embedding_shape_and_dtype() -> None:
    time = torch.tensor(
        [0.0, 0.5, 1.0],
        dtype=torch.float32,
    )

    embedding = create_sinusoidal_pos_embedding(
        time,
        dimension=8,
    )

    assert embedding.shape == (3, 8)
    assert embedding.dtype == torch.float32
    assert embedding.device == time.device


def test_zero_time_has_zero_sine_and_one_cosine() -> None:
    time = torch.tensor(
        [0.0],
        dtype=torch.float32,
    )

    embedding = create_sinusoidal_pos_embedding(
        time,
        dimension=8,
    )

    sine_half = embedding[:, :4]
    cosine_half = embedding[:, 4:]

    assert torch.allclose(
        sine_half,
        torch.zeros_like(sine_half),
    )
    assert torch.allclose(
        cosine_half,
        torch.ones_like(cosine_half),
    )


def test_different_times_have_different_embeddings() -> None:
    time = torch.tensor(
        [0.1, 0.9],
        dtype=torch.float32,
    )

    embedding = create_sinusoidal_pos_embedding(
        time,
        dimension=8,
    )

    assert not torch.allclose(
        embedding[0],
        embedding[1],
    )


def test_odd_dimension_is_rejected() -> None:
    time = torch.tensor(
        [0.5],
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="even"):
        create_sinusoidal_pos_embedding(
            time,
            dimension=7,
        )


def test_non_vector_time_is_rejected() -> None:
    time = torch.zeros(
        2,
        1,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match=r"\[B\]"):
        create_sinusoidal_pos_embedding(
            time,
            dimension=8,
        )


def test_integer_time_is_rejected() -> None:
    time = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    with pytest.raises(TypeError, match="floating point"):
        create_sinusoidal_pos_embedding(
            time,
            dimension=8,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_cuda_output_stays_on_cuda() -> None:
    time = torch.tensor(
        [0.0, 0.5, 1.0],
        dtype=torch.float32,
        device="cuda",
    )

    embedding = create_sinusoidal_pos_embedding(
        time,
        dimension=8,
    )

    assert embedding.device.type == "cuda"
    assert embedding.dtype == torch.float32
