import torch

from pi0.normalization import (
    NormStats,
    Pi0Normalizer,
    compute_norm_stats,
    load_norm_stats,
    save_norm_stats,
)


def make_stats() -> NormStats:
    return NormStats(
        mean=torch.tensor([10.0, 20.0]),
        std=torch.tensor([2.0, 4.0]),
        q01=torch.tensor([0.0, 10.0]),
        q99=torch.tensor([20.0, 30.0]),
    )


def test_standard_normalization_round_trip() -> None:
    stats = make_stats()

    normalizer = Pi0Normalizer(
        state_stats=stats,
        action_stats=stats,
    )

    values = torch.tensor([[12.0, 24.0]])

    normalized = normalizer.normalize_actions(values)
    restored = normalizer.unnormalize_actions(normalized)

    assert torch.allclose(
        normalized,
        torch.tensor([[1.0, 1.0]]),
        atol=1e-5,
    )
    assert torch.allclose(
        restored,
        values,
        atol=1e-5,
    )


def test_quantile_normalization_round_trip() -> None:
    stats = make_stats()

    normalizer = Pi0Normalizer(
        state_stats=stats,
        action_stats=stats,
        use_quantiles=True,
    )

    values = torch.tensor([[0.0, 30.0]])

    normalized = normalizer.normalize_actions(values)
    restored = normalizer.unnormalize_actions(normalized)

    assert torch.allclose(
        normalized,
        torch.tensor([[-1.0, 1.0]]),
        atol=1e-5,
    )
    assert torch.allclose(
        restored,
        values,
        atol=1e-5,
    )


def test_padding_dimensions_remain_unchanged() -> None:
    stats = make_stats()

    normalizer = Pi0Normalizer(
        state_stats=stats,
        action_stats=stats,
    )

    values = torch.tensor([[12.0, 24.0, 0.0, 0.0]])

    normalized = normalizer.normalize_state(values)

    assert torch.allclose(
        normalized,
        torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        atol=1e-5,
    )


def test_compute_statistics() -> None:
    values = torch.tensor(
        [
            [0.0, 10.0],
            [2.0, 20.0],
            [4.0, 30.0],
        ]
    )

    stats = compute_norm_stats(values)

    assert torch.allclose(
        stats.mean,
        torch.tensor([2.0, 20.0]),
    )
    assert torch.allclose(
        stats.std,
        values.std(dim=0, correction=0),
    )


def test_statistics_json_round_trip(tmp_path) -> None:
    path = tmp_path / "norm_stats.json"

    save_norm_stats(
        path,
        {
            "state": make_stats(),
            "actions": make_stats(),
        },
    )

    restored = load_norm_stats(path)

    assert set(restored) == {
        "state",
        "actions",
    }
    assert torch.equal(
        restored["state"].mean,
        make_stats().mean,
    )
    assert torch.equal(
        restored["actions"].q99,
        make_stats().q99,
    )
