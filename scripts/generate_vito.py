from __future__ import annotations

import argparse
from pathlib import Path

from inference.generation import generate


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with VITO."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "experiments/vito-v0.2/checkpoint-500"
        ),
    )

    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(
            "tokenizer/vito-bpe-v0.2.json"
        ),
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    print("=" * 64)
    print("VITO GENERATION")
    print("=" * 64)
    print("Checkpoint:", args.checkpoint)
    print("Tokenizer:", args.tokenizer)
    print("Prompt:", args.prompt)
    print()

    # The current generation implementation uses its own defaults,
    # so temporarily expose the selected checkpoint/tokenizer through
    # the generation module.
    import inference.generation as generation_module

    model, tokenizer, device = generation_module.load_vito(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
    )

    import torch

    input_ids = tokenizer(
        args.prompt,
        return_tensors="pt",
    )["input_ids"].to(device)

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            if (
                input_ids.shape[1]
                > model.config.max_position_embeddings
            ):
                input_ids = input_ids[
                    :,
                    -model.config.max_position_embeddings:,
                ]

            outputs = model(
                input_ids=input_ids
            )

            logits = outputs.logits[:, -1, :]
            logits = logits / max(
                args.temperature,
                1e-5,
            )

            if args.top_k is not None:
                top_k = min(
                    args.top_k,
                    logits.size(-1),
                )

                values, indices = torch.topk(
                    logits,
                    top_k,
                )

                filtered_logits = torch.full_like(
                    logits,
                    float("-inf"),
                )

                filtered_logits.scatter_(
                    1,
                    indices,
                    values,
                )

                logits = filtered_logits

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1,
            )

    generated = tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True,
    )

    print("Device:", device)
    print()
    print("-" * 64)
    print(generated)
    print("-" * 64)


if __name__ == "__main__":
    main()