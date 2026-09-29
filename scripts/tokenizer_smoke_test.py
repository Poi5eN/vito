#!/usr/bin/env python3
"""Inspect a trained tokenizer on representative developer and venture text."""
from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer

SAMPLES = [
    "Fix the Node.js API error and verify the database failure.",
    "Mera React component initial render par crash ho raha hai.",
    "Calculate LTV:CAC and reconcile it with source data.",
    "Founder claims 120% NRR; verify it from cohort data.",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", type=Path)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    for sample in SAMPLES:
        encoding = tokenizer.encode(sample)
        print("\nTEXT:", sample)
        print("TOKENS:", encoding.tokens)
        print("IDS:", encoding.ids)
        print("TOKEN COUNT:", len(encoding.ids))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
