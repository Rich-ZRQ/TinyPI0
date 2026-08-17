"""Inspect the two capacities of the same Tiny pi0 architecture."""

from configs import SO101_TINY, TINY_PI0


def main() -> None:
    print("Tiny pi0 debug capacity:")
    print(TINY_PI0)
    print("Visual tokens per image:", TINY_PI0.vision.num_tokens)

    print()

    print("Tiny pi0 SO101 training capacity:")
    print(SO101_TINY)
    print("Visual tokens per image:", SO101_TINY.vision.num_tokens)


if __name__ == "__main__":
    main()
