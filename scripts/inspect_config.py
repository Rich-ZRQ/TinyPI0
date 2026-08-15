"""Inspect development, SO-ARM101 and official-size configurations."""

from configs import (
    FULL_PI0,
    LOCAL_RUNTIME,
    SERVER_RUNTIME,
    SO101_LARGE,
    SO101_LOCAL_RUNTIME,
    SO101_RECOMMENDED,
    SO101_SERVER_RUNTIME,
    TINY_PI0,
)


def main() -> None:
    print("Tiny pi0:")
    print(TINY_PI0)
    print("Tiny visual tokens:", TINY_PI0.vision.num_tokens)
    print("Tiny runtime:", LOCAL_RUNTIME)

    print()

    print("SO101 recommended (real local inference / training baseline):")
    print(SO101_RECOMMENDED)
    print("SO101 visual tokens:", SO101_RECOMMENDED.vision.num_tokens)
    print("SO101 local runtime:", SO101_LOCAL_RUNTIME)
    print("SO101 server runtime:", SO101_SERVER_RUNTIME)

    print()

    print("SO101 large (validation ablation):")
    print(SO101_LARGE)

    print()

    print("Full pi0:")
    print(FULL_PI0)
    print("Full visual tokens:", FULL_PI0.vision.num_tokens)
    print("Full runtime:", SERVER_RUNTIME)


if __name__ == "__main__":
    main()
