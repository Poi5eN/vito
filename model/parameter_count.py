#!/usr/bin/env python3
"""Print the exact parameter count implied by the VITO decoder architecture."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def count_parameters(config: dict, tie_embeddings: bool = True) -> int:
    m = config["model"]
    vocab = int(m["vocab_size"])
    hidden = int(m["hidden_size"])
    layers = int(m["num_layers"])
    intermediate = int(m["intermediate_size"])

    # Token embedding: vocab x hidden.
    total = vocab * hidden

    # Learned final LayerNorm: gamma + beta.
    total += 2 * hidden

    for _ in range(layers):
        # Pre-attention LayerNorm.
        total += 2 * hidden
        # Q, K, V projections + output projection.
        total += 4 * hidden * hidden + 4 * hidden
        # Pre-MLP LayerNorm.
        total += 2 * hidden
        # Two-layer MLP with biases.
        total += 2 * hidden * intermediate + intermediate + hidden

    # LM head.
    if not tie_embeddings:
        total += vocab * hidden + vocab
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--untied-embeddings", action="store_true")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    params = count_parameters(config, tie_embeddings=not args.untied_embeddings)
    print(f"Parameters: {params:,}")
    print(f"Parameters (millions): {params / 1_000_000:.3f}M")
    print(f"Embedding/LM-head tied: {not args.untied_embeddings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
