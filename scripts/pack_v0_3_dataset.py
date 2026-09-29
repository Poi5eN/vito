from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pack VITO v0.3 JSONL records into fixed-length token sequences."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "data/processed/vito_corpus_v0.3",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "tokenizer/vito-bpe-v0.3.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/processed/vito_packed_v0.3",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}:{line_number}"
                ) from exc

            records.append(record)

    return records


def record_to_text(record: dict) -> str:
    """
    Convert a normalized VITO v0.3 record into a deterministic
    training representation.

    We deliberately preserve the task/context/output structure so
    the model learns VITO's verification-oriented format.
    """

    parts = []

    domain = record.get("domain")
    record_type = record.get("record_type")
    input_text = record.get("input", "")
    context = record.get("context", "")
    output = record.get("output", "")

    if domain:
        parts.append(f"Domain: {domain}")

    if record_type:
        parts.append(f"Record type: {record_type}")

    if input_text:
        parts.append(f"Input:\n{input_text}")

    if context:
        parts.append(f"Context:\n{context}")

    parts.append(f"Output:\n{output}")

    return "\n\n".join(parts).strip()


def pack_split(
    records: list[dict],
    tokenizer: Tokenizer,
    sequence_length: int,
    seed: int,
):
    rng = random.Random(seed)

    # Shuffle records deterministically before packing.
    records = list(records)
    rng.shuffle(records)

    eos_id = tokenizer.token_to_id("<eos>")

    if eos_id is None:
        raise ValueError(
            "Could not find the required <eos> token in the tokenizer."
        )

    token_stream: list[int] = []
    document_lengths: list[int] = []

    for record in records:
        text = record_to_text(record)

        if not text:
            continue

        encoded = tokenizer.encode(text)
        ids = encoded.ids

        if not ids:
            continue

        ids = ids + [eos_id]

        token_stream.extend(ids)
        document_lengths.append(len(ids))

    if len(token_stream) < sequence_length + 1:
        raise ValueError(
            f"Not enough tokens to create sequences of length {sequence_length}."
        )

    # We need sequence_length + 1 tokens per training example:
    #
    # input  = tokens[:-1]
    # labels = tokens[1:]
    #
    usable_tokens = (
        len(token_stream) - 1
    ) // sequence_length * sequence_length + 1

    token_stream = token_stream[:usable_tokens]

    examples = []

    for start in range(0, len(token_stream) - 1, sequence_length):
        chunk = token_stream[start:start + sequence_length + 1]

        if len(chunk) != sequence_length + 1:
            break

        examples.append(
            {
                "input_ids": chunk[:-1],
                "labels": chunk[1:],
            }
        )

    return examples, len(token_stream), document_lengths


def save_split(examples: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example) + "\n")


def main():
    args = parse_args()

    if args.sequence_length < 2:
        raise ValueError("sequence-length must be at least 2")

    if not args.tokenizer.exists():
        raise FileNotFoundError(args.tokenizer)

    if not args.input_dir.exists():
        raise FileNotFoundError(args.input_dir)

    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    args.output_dir.mkdir(parents=True)

    manifest = {
        "version": "0.3.0",
        "tokenizer": str(args.tokenizer),
        "vocab_size": tokenizer.get_vocab_size(),
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "splits": {},
    }

    print("=" * 64)
    print("VITO v0.3 DATASET PACKING")
    print("=" * 64)
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Vocabulary: {tokenizer.get_vocab_size()}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Seed: {args.seed}")
    print()

    for split in ("train", "validation", "test"):
        source = args.input_dir / f"{split}.jsonl"

        if not source.exists():
            raise FileNotFoundError(source)

        records = load_jsonl(source)

        examples, token_count, document_lengths = pack_split(
            records=records,
            tokenizer=tokenizer,
            sequence_length=args.sequence_length,
            seed=args.seed,
        )

        output = args.output_dir / f"{split}.jsonl"

        save_split(examples, output)

        discarded = token_count - len(examples) * args.sequence_length

        stats = {
            "records": len(records),
            "packed_sequences": len(examples),
            "tokens_consumed": token_count,
            "discarded_boundary_tokens": discarded,
            "average_document_tokens": (
                sum(document_lengths) / len(document_lengths)
                if document_lengths
                else 0
            ),
            "min_document_tokens": min(document_lengths)
            if document_lengths
            else 0,
            "max_document_tokens": max(document_lengths)
            if document_lengths
            else 0,
        }

        manifest["splits"][split] = stats

        print("-" * 64)
        print(split.upper())
        print("-" * 64)
        print(f"Records:              {stats['records']:,}")
        print(f"Packed sequences:     {stats['packed_sequences']:,}")
        print(f"Tokens consumed:      {stats['tokens_consumed']:,}")
        print(f"Discarded tokens:     {stats['discarded_boundary_tokens']:,}")
        print(
            f"Average document:     "
            f"{stats['average_document_tokens']:.2f}"
        )
        print(f"Minimum document:     {stats['min_document_tokens']}")
        print(f"Maximum document:     {stats['max_document_tokens']}")

    manifest_path = args.output_dir / "manifest.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 64)
    print("VITO v0.3 PACKING COMPLETE")
    print("=" * 64)
    print(f"Output: {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
