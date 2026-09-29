#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer

from model.configuration_vito import VitoConfig
from model.modeling_vito import VitoForCausalLM


ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_FILE = (
    ROOT / "tokenizer" / "vito-bpe-v0.2.json"
)

OUTPUT_DIR = (
    ROOT / "experiments" / "vito-v0.2" / "model"
)


def main() -> None:

    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_FILE}"
        )

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_FILE)
    )

    vocab_size = tokenizer.get_vocab_size()

    print("=" * 60)
    print("BUILDING VITO v0.2 MODEL")
    print("=" * 60)
    print(f"Tokenizer: {TOKENIZER_FILE}")
    print(f"Vocabulary size: {vocab_size}")

    config = VitoConfig(
        vocab_size=vocab_size,
        max_position_embeddings=512,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        tie_word_embeddings=True,
    )

    model = VitoForCausalLM(config)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config.save_pretrained(
        OUTPUT_DIR
    )

    metadata = {
        "version": "0.2.0",
        "tokenizer": str(
            TOKENIZER_FILE.relative_to(ROOT)
        ),
        "vocab_size": vocab_size,
        "parameters": parameter_count,
        "trainable_parameters": trainable_parameter_count,
        "architecture": {
            "context_length": 512,
            "hidden_size": 512,
            "layers": 8,
            "heads": 8,
            "intermediate_size": 2048,
            "tie_word_embeddings": True,
        },
    }

    metadata_file = OUTPUT_DIR / "model_metadata.json"

    metadata_file.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Parameters: {parameter_count:,}")
    print(
        "Trainable parameters: "
        f"{trainable_parameter_count:,}"
    )
    print(f"Saved config: {OUTPUT_DIR}")
    print(f"Metadata: {metadata_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
