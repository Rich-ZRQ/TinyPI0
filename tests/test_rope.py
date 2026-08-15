import pytest
import torch

from pi0.rope import (
    GemmaRotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)


def test_rotate_half() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])

    actual = rotate_half(x)

    expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])

    assert torch.equal(actual, expected)


def test_rotary_embedding_shape() -> None:
    rope = GemmaRotaryEmbedding(head_dim=8)

    x = torch.zeros(
        2,
        3,
        8,
        dtype=torch.float32,
    )
    position_ids = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, 5],
        ],
        dtype=torch.long,
    )

    cosine, sine = rope(x, position_ids)

    assert cosine.shape == (2, 3, 8)
    assert sine.shape == (2, 3, 8)
    assert cosine.dtype == x.dtype
    assert sine.dtype == x.dtype


def test_position_zero_is_identity() -> None:
    rope = GemmaRotaryEmbedding(head_dim=8)

    reference = torch.zeros(
        1,
        1,
        8,
    )
    position_ids = torch.zeros(
        1,
        1,
        dtype=torch.long,
    )

    cosine, sine = rope(
        reference,
        position_ids,
    )

    assert torch.equal(
        cosine,
        torch.ones_like(cosine),
    )
    assert torch.equal(
        sine,
        torch.zeros_like(sine),
    )


def test_rope_at_position_zero_does_not_change_qk() -> None:
    rope = GemmaRotaryEmbedding(head_dim=8)

    query = torch.randn(1, 4, 1, 8)
    key = torch.randn(1, 1, 1, 8)

    position_ids = torch.zeros(
        1,
        1,
        dtype=torch.long,
    )

    reference = query[:, 0]
    cosine, sine = rope(
        reference,
        position_ids,
    )

    rotated_query, rotated_key = apply_rotary_pos_emb(
        query,
        key,
        cosine,
        sine,
    )

    assert torch.equal(rotated_query, query)
    assert torch.equal(rotated_key, key)


def test_rotation_preserves_vector_norm() -> None:
    rope = GemmaRotaryEmbedding(head_dim=8)

    query = torch.randn(2, 4, 5, 8)
    key = torch.randn(2, 1, 5, 8)

    position_ids = torch.arange(5).repeat(2, 1)

    reference = query[:, 0]
    cosine, sine = rope(
        reference,
        position_ids,
    )

    rotated_query, rotated_key = apply_rotary_pos_emb(
        query,
        key,
        cosine,
        sine,
    )

    original_query_norm = torch.linalg.vector_norm(
        query,
        dim=-1,
    )
    rotated_query_norm = torch.linalg.vector_norm(
        rotated_query,
        dim=-1,
    )

    original_key_norm = torch.linalg.vector_norm(
        key,
        dim=-1,
    )
    rotated_key_norm = torch.linalg.vector_norm(
        rotated_key,
        dim=-1,
    )

    assert torch.allclose(
        original_query_norm,
        rotated_query_norm,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        original_key_norm,
        rotated_key_norm,
        atol=1e-5,
        rtol=1e-5,
    )


def test_gradient_flows_through_rotation() -> None:
    rope = GemmaRotaryEmbedding(head_dim=8)

    query = torch.randn(
        1,
        4,
        3,
        8,
        requires_grad=True,
    )
    key = torch.randn(
        1,
        1,
        3,
        8,
        requires_grad=True,
    )
    position_ids = torch.arange(3)[None, :]

    reference = query[:, 0]
    cosine, sine = rope(
        reference,
        position_ids,
    )

    rotated_query, rotated_key = apply_rotary_pos_emb(
        query,
        key,
        cosine,
        sine,
    )

    loss = rotated_query.square().mean() + rotated_key.square().mean()
    loss.backward()

    assert query.grad is not None
    assert key.grad is not None
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(key.grad).all()


def test_odd_head_dimension_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must be even",
    ):
        GemmaRotaryEmbedding(head_dim=7)


def test_invalid_position_shape_is_rejected() -> None:
    rope = GemmaRotaryEmbedding(head_dim=8)

    x = torch.zeros(2, 3, 8)
    position_ids = torch.zeros(
        2,
        3,
        1,
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match=r"\[B, S\]",
    ):
        rope(x, position_ids)
