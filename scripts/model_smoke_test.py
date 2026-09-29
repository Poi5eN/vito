import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import VitoConfig, VitoForCausalLM


def main():
    print("=" * 70)
    print("VITO Transformer Smoke Test")
    print("=" * 70)

    tokenizer_path = ROOT / "tokenizer" / "vito-bpe-v0.1.json"

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}"
        )

    tokenizer = Tokenizer.from_file(
        str(tokenizer_path)
    )

    vocab_size = tokenizer.get_vocab_size()

    print(f"Tokenizer vocabulary: {vocab_size}")

    config = VitoConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
    )

    model = VitoForCausalLM(config)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Model parameters: "
        f"{parameter_count:,} "
        f"({parameter_count / 1_000_000:.3f}M)"
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    model = model.to(device)

    text = (
        "Fix the Node.js API error "
        "and verify the database failure."
    )

    encoded = tokenizer.encode(text)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device,
    )

    print(f"Input text: {text}")
    print(f"Token count: {len(encoded.ids)}")
    print(f"Input tensor shape: {tuple(input_ids.shape)}")

    model.train()

    outputs = model(
        input_ids=input_ids,
        labels=input_ids,
        output_attentions=True,
        output_hidden_states=True,
    )

    print(f"Logits shape: {tuple(outputs.logits.shape)}")

    if outputs.loss is None:
        raise RuntimeError("Loss was not produced.")

    print(f"Loss: {outputs.loss.item():.6f}")

    print(
        f"Attention layers: "
        f"{len(outputs.attentions)}"
    )

    print(
        f"Hidden-state layers: "
        f"{len(outputs.hidden_states)}"
    )

    outputs.loss.backward()

    print("Backward pass: PASS")

    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight

    tied = (
        input_weight.data_ptr()
        == output_weight.data_ptr()
    )

    print(
        f"Weight tying: "
        f"{'PASS' if tied else 'FAIL'}"
    )

    if not tied:
        raise RuntimeError(
            "Input embeddings and LM head are not tied."
        )

    print("=" * 70)
    print("VITO TRANSFORMER SMOKE TEST: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
