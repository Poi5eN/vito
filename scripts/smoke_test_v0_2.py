#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import torch

from tokenizers import Tokenizer

from model.configuration_vito import VitoConfig
from model.modeling_vito import VitoForCausalLM


ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_FILE = (
    ROOT / "tokenizer" / "vito-bpe-v0.2.json"
)

CONFIG_DIR = (
    ROOT / "experiments" / "vito-v0.2" / "model"
)


def get_device() -> torch.device:

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def main() -> None:

    device = get_device()

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_FILE)
    )

    config = VitoConfig.from_pretrained(
        CONFIG_DIR
    )

    model = VitoForCausalLM(config)
    model.to(device)
    model.eval()

    text = (
        "Fix the database error and "
        "verify the solution."
    )

    encoded = tokenizer.encode(text)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device,
    )

    labels = input_ids.clone()

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            labels=labels,
        )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("=" * 60)
    print("VITO v0.2 MODEL SMOKE TEST")
    print("=" * 60)
    print(f"Device: {device}")
    print(
        f"Vocabulary: "
        f"{tokenizer.get_vocab_size()}"
    )
    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )
    print(
        f"Input shape: "
        f"{tuple(input_ids.shape)}"
    )
    print(
        f"Logits shape: "
        f"{tuple(outputs.logits.shape)}"
    )
    print(
        f"Loss: "
        f"{outputs.loss.item():.6f}"
    )

    assert (
        outputs.logits.shape[0]
        == input_ids.shape[0]
    )

    assert (
        outputs.logits.shape[1]
        == input_ids.shape[1]
    )

    assert (
        outputs.logits.shape[2]
        == tokenizer.get_vocab_size()
    )

    assert torch.isfinite(
        outputs.loss
    )

    print()
    print("MODEL SMOKE TEST: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
