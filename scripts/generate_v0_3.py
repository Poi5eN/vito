from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import VitoConfig, VitoForCausalLM


ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_PATH = ROOT / "tokenizer/vito-bpe-v0.3.json"
CHECKPOINT = ROOT / "experiments/vito-v0.3/checkpoint-100"

MAX_NEW_TOKENS = 120
TEMPERATURE = 0.8
TOP_K = 50


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


PROMPTS = [
    (
        "DEVELOPER",
        """How do I debug a Node.js HTTP server that is returning an unexpected response?

Explain the verification steps."""
    ),
    (
        "SQL",
        """Write a SQL query to find duplicate email addresses.

Explain how to verify the result."""
    ),
    (
        "VERIFICATION",
        """An API returns HTTP 200 but the application data appears incorrect.

How should I verify the response?"""
    ),
    (
        "VENTURE",
        """A startup has revenue of $100,000 in January and $130,000 in February.

Calculate the monthly growth rate and explain how to verify the calculation."""
    ),
    (
        "GENERAL",
        """What is the difference between an assumption and verified evidence?

Give a structured explanation."""
    ),
]


def sample_next_token(logits: torch.Tensor) -> torch.Tensor:
    """
    Sample one token using temperature + top-k sampling.
    """

    logits = logits / TEMPERATURE

    if TOP_K is not None and TOP_K > 0:
        k = min(TOP_K, logits.size(-1))

        values, indices = torch.topk(logits, k=k)

        filtered_logits = torch.full_like(
            logits,
            float("-inf"),
        )

        filtered_logits.scatter_(
            -1,
            indices,
            values,
        )

        logits = filtered_logits

    probabilities = torch.softmax(logits, dim=-1)

    next_token = torch.multinomial(
        probabilities,
        num_samples=1,
    )

    return next_token


def generate(
    model: VitoForCausalLM,
    input_ids: torch.Tensor,
    eos_token_id: int | None,
    max_new_tokens: int,
) -> torch.Tensor:

    generated = input_ids.clone()

    for _ in range(max_new_tokens):

        # Keep the context within the model's configured context window.
        max_context = model.config.max_position_embeddings

        context = generated[:, -max_context:]

        with torch.no_grad():
            outputs = model(
                input_ids=context,
            )

        logits = outputs.logits[:, -1, :]

        next_token = sample_next_token(logits)

        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

        if eos_token_id is not None:
            if torch.all(next_token == eos_token_id):
                break

    return generated


def main():
    print("=" * 72)
    print("VITO v0.3 GENERATION TEST")
    print("=" * 72)

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    config = VitoConfig.from_pretrained(
        str(CHECKPOINT)
    )

    model = VitoForCausalLM.from_pretrained(
        str(CHECKPOINT),
        config=config,
    )

    device = torch.device(
        detect_device()
    )

    model.to(device)
    model.eval()

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Vocabulary: {tokenizer.get_vocab_size()}")
    print(f"Device: {device}")
    print(
        f"Parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )
    print()

    # ---------------------------------------------------------------
    # Verify tied embeddings
    # ---------------------------------------------------------------

    tied = (
        model.lm_head.weight.data_ptr()
        == model.vito.token_embeddings.weight.data_ptr()
    )

    print(f"LM head tied to token embeddings: {tied}")

    if not tied:
        raise RuntimeError(
            "VITO weight tying is not active after checkpoint loading."
        )

    print()

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    print(f"BOS token ID: {bos_id}")
    print(f"EOS token ID: {eos_id}")
    print()

    # ---------------------------------------------------------------
    # Generate
    # ---------------------------------------------------------------

    for category, prompt in PROMPTS:

        print("=" * 72)
        print(category)
        print("=" * 72)

        print("PROMPT:")
        print(prompt)
        print()

        encoded = tokenizer.encode(prompt)

        prompt_ids = encoded.ids

        if bos_id is not None:
            prompt_ids = [bos_id] + prompt_ids

        input_tensor = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=device,
        )

        generated = generate(
            model=model,
            input_ids=input_tensor,
            eos_token_id=eos_id,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        generated_ids = (
            generated[0]
            .detach()
            .cpu()
            .tolist()
        )

        generated_only = generated_ids[
            len(prompt_ids):
        ]

        text = tokenizer.decode(
            generated_only,
            skip_special_tokens=True,
        )

        print("GENERATED:")
        print(text.strip())
        print()


if __name__ == "__main__":
    main()
