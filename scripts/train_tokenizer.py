#!/usr/bin/env python3

"""Train VITO's custom byte-level BPE tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)


SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<domain>",
    "</domain>",
    "<task>",
    "</task>",
    "<language>",
    "</language>",
    "<input>",
    "</input>",
    "<context>",
    "</context>",
    "<output>",
    "</output>",
    "<problem>",
    "</problem>",
    "<hypothesis>",
    "</hypothesis>",
    "<verify>",
    "</verify>",
    "<action>",
    "</action>",
    "<verification>",
    "</verification>",
    "<evidence>",
    "</evidence>",
]


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-file",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "tokenizer/vito-bpe-v0.1.json"
        ),
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=12288,
    )

    args = parser.parse_args()

    if not args.train_file.exists():
        raise FileNotFoundError(
            args.train_file
        )

    if args.vocab_size < len(SPECIAL_TOKENS):
        raise ValueError(
            "vocab-size must be at least "
            f"{len(SPECIAL_TOKENS)}."
        )

    tokenizer = Tokenizer(
        models.BPE(
            unk_token="<unk>"
        )
    )

    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.NFKC()
        ]
    )

    tokenizer.pre_tokenizer = (
        pre_tokenizers.ByteLevel(
            add_prefix_space=False
        )
    )

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train(
        [str(args.train_file)],
        trainer,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save(
        str(args.output)
    )

    print(
        f"Saved tokenizer: {args.output}"
    )

    print(
        "Vocabulary size: "
        f"{tokenizer.get_vocab_size()}"
    )

    print(
        "Special tokens: "
        f"{len(SPECIAL_TOKENS)}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())