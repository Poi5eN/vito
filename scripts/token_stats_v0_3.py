#!/usr/bin/env python3

"""
Calculate exact token statistics for the VITO v0.3 corpus.

Uses the trained VITO v0.3 tokenizer and evaluates each
processed split independently.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]

CORPUS_DIR = ROOT / "data" / "processed" / "vito_corpus_v0.3"
TOKENIZER_PATH = ROOT / "tokenizer" / "vito-bpe-v0.3.json"

SPLITS = ("train", "validation", "test")


def record_text(record: dict) -> str:
    """
    Convert a normalized VITO record into the text representation
    used for tokenizer statistics.

    We preserve the structured fields instead of counting only
    `output`.
    """

    parts = []

    for field in ("input", "context", "output"):
        value = record.get(field)

        if value is None:
            continue

        value = str(value).strip()

        if value:
            parts.append(value)

    return "\n".join(parts)


def analyze_split(tokenizer: Tokenizer, split: str) -> dict:

    path = CORPUS_DIR / f"{split}.jsonl"

    records = 0
    characters = 0
    tokens = 0

    min_tokens = None
    max_tokens = None

    with path.open("r", encoding="utf-8") as handle:

        for line in handle:

            if not line.strip():
                continue

            record = json.loads(line)

            text = record_text(record)

            encoding = tokenizer.encode(text)

            token_count = len(encoding.ids)

            records += 1
            characters += len(text)
            tokens += token_count

            if min_tokens is None or token_count < min_tokens:
                min_tokens = token_count

            if max_tokens is None or token_count > max_tokens:
                max_tokens = token_count

    average_tokens = (
        tokens / records
        if records
        else 0
    )

    tokens_per_character = (
        tokens / characters
        if characters
        else 0
    )

    return {
        "records": records,
        "characters": characters,
        "tokens": tokens,
        "average_tokens_per_record": average_tokens,
        "tokens_per_character": tokens_per_character,
        "min_tokens": min_tokens or 0,
        "max_tokens": max_tokens or 0,
    }


def main() -> None:

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_PATH}"
        )

    if not CORPUS_DIR.exists():
        raise FileNotFoundError(
            f"Corpus not found: {CORPUS_DIR}"
        )

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    vocab_size = tokenizer.get_vocab_size()

    print("=" * 68)
    print("VITO v0.3 EXACT TOKEN STATISTICS")
    print("=" * 68)

    print()
    print(f"Tokenizer: {TOKENIZER_PATH}")
    print(f"Vocabulary size: {vocab_size:,}")

    results = {}

    for split in SPLITS:

        result = analyze_split(
            tokenizer,
            split,
        )

        results[split] = result

        print()
        print("-" * 68)
        print(split.upper())
        print("-" * 68)

        print(
            f"Records:                {result['records']:,}"
        )

        print(
            f"Characters:             {result['characters']:,}"
        )

        print(
            f"Exact tokens:           {result['tokens']:,}"
        )

        print(
            "Average tokens/record:  "
            f"{result['average_tokens_per_record']:.2f}"
        )

        print(
            "Tokens/character:       "
            f"{result['tokens_per_character']:.4f}"
        )

        print(
            f"Minimum tokens/record:  {result['min_tokens']:,}"
        )

        print(
            f"Maximum tokens/record:  {result['max_tokens']:,}"
        )

    total_records = sum(
        result["records"]
        for result in results.values()
    )

    total_characters = sum(
        result["characters"]
        for result in results.values()
    )

    total_tokens = sum(
        result["tokens"]
        for result in results.values()
    )

    print()
    print("=" * 68)
    print("TOTAL")
    print("=" * 68)

    print(
        f"Records:                {total_records:,}"
    )

    print(
        f"Characters:             {total_characters:,}"
    )

    print(
        f"Exact tokens:           {total_tokens:,}"
    )

    print(
        "Average tokens/record:  "
        f"{total_tokens / total_records:.2f}"
    )

    print(
        "Tokens/character:       "
        f"{total_tokens / total_characters:.4f}"
    )

    print()
    print("=" * 68)


if __name__ == "__main__":
    main()
