import pytest
import torch

from pi0.attention_mask import make_att_2d_masks


def test_pure_causal_attention() -> None:
    pad_masks = torch.ones(
        1,
        4,
        dtype=torch.bool,
    )
    att_masks = torch.ones(
        1,
        4,
        dtype=torch.bool,
    )

    actual = make_att_2d_masks(
        pad_masks,
        att_masks,
    )

    expected = torch.tensor(
        [
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        ]
    )

    assert torch.equal(actual, expected)


def test_prefix_lm_attention() -> None:
    pad_masks = torch.ones(
        1,
        6,
        dtype=torch.bool,
    )
    att_masks = torch.tensor(
        [
            [
                False,
                False,
                False,
                True,
                True,
                True,
            ]
        ]
    )

    actual = make_att_2d_masks(
        pad_masks,
        att_masks,
    )

    expected = torch.tensor(
        [
            [
                [True, True, True, False, False, False],
                [True, True, True, False, False, False],
                [True, True, True, False, False, False],
                [True, True, True, True, False, False],
                [True, True, True, True, True, False],
                [True, True, True, True, True, True],
            ]
        ]
    )

    assert torch.equal(actual, expected)


def test_block_causal_attention() -> None:
    pad_masks = torch.ones(
        1,
        4,
        dtype=torch.bool,
    )
    att_masks = torch.tensor(
        [
            [
                True,
                False,
                True,
                False,
            ]
        ]
    )

    actual = make_att_2d_masks(
        pad_masks,
        att_masks,
    )

    expected = torch.tensor(
        [
            [
                [True, True, False, False],
                [True, True, False, False],
                [True, True, True, True],
                [True, True, True, True],
            ]
        ]
    )

    assert torch.equal(actual, expected)


def test_padding_masks_query_and_key() -> None:
    pad_masks = torch.tensor(
        [
            [
                True,
                True,
                False,
            ]
        ]
    )
    att_masks = torch.ones(
        1,
        3,
        dtype=torch.bool,
    )

    actual = make_att_2d_masks(
        pad_masks,
        att_masks,
    )

    expected = torch.tensor(
        [
            [
                [True, False, False],
                [True, True, False],
                [False, False, False],
            ]
        ]
    )

    assert torch.equal(actual, expected)


def test_shape_mismatch_is_rejected() -> None:
    pad_masks = torch.ones(
        2,
        4,
        dtype=torch.bool,
    )
    att_masks = torch.ones(
        2,
        3,
        dtype=torch.bool,
    )

    with pytest.raises(ValueError, match="same shape"):
        make_att_2d_masks(
            pad_masks,
            att_masks,
        )


def test_non_boolean_mask_is_rejected() -> None:
    pad_masks = torch.ones(
        1,
        4,
        dtype=torch.float32,
    )
    att_masks = torch.ones(
        1,
        4,
        dtype=torch.bool,
    )

    with pytest.raises(TypeError, match="pad_masks must be bool"):
        make_att_2d_masks(
            pad_masks,
            att_masks,
        )
