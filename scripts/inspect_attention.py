import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import VitoConfig, VitoForCausalLM


def main():
    print("=" * 70)
    print("VITO Attention Inspection")
    print("=" * 70)

    tokenizer = Tokenizer.from_file(
        str(ROOT / "tokenizer" / "vito-bpe-v0.1.json")
    )

    config = VitoConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=128,
    )

    model = VitoForCausalLM(config)

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    model.to(device)
    model.eval()

    text = "Verify the database error."

    encoded = tokenizer.encode(text)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device,
    )

    print(f"Text: {text}")
    print(f"Tokens: {encoded.tokens}")
    print(f"Token IDs: {encoded.ids}")
    print(f"Device: {device}")

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_attentions=True,
        )

    attention = outputs.attentions[0]

    print()
    print(
        "Layer 0 attention shape:",
        tuple(attention.shape),
    )

    # First batch, first attention head.
    head = attention[0, 0].detach().cpu()

    print()
    print("Head 0 attention matrix:")
    print(head)

    # Check causal property.
    future_attention = torch.triu(
        head,
        diagonal=1,
    )

    max_future_attention = future_attention.abs().max().item()

    print()
    print(
        "Maximum attention to future tokens:",
        f"{max_future_attention:.8f}",
    )

    if max_future_attention > 1e-5:
        raise RuntimeError(
            "Causal attention check FAILED."
        )

    print("Causal attention check: PASS")

    print("=" * 70)
    print("VITO ATTENTION INSPECTION: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
