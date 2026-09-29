from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

from model import VitoConfig, VitoForCausalLM


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINT = (
    ROOT
    / "experiments"
    / "vito-v0.2"
    / "checkpoint-500"
)

DEFAULT_TOKENIZER = (
    ROOT
    / "tokenizer"
    / "vito-bpe-v0.2.json"
)

OUTPUT_FILE = (
    ROOT
    / "experiments"
    / "vito-v0.2"
    / "benchmark_v0.1.json"
)


PROMPTS = [
    {
        "id": "dev_debug_001",
        "domain": "developer",
        "task": "debugging",
        "prompt": (
            "A Node.js API crashes when connecting to PostgreSQL. "
            "Explain what should be checked first."
        ),
    },
    {
        "id": "dev_debug_002",
        "domain": "developer",
        "task": "debugging",
        "prompt": (
            "A React component crashes during its initial render. "
            "Give a systematic debugging approach."
        ),
    },
    {
        "id": "dev_sql_001",
        "domain": "developer",
        "task": "sql",
        "prompt": (
            "Write a SQL query to find users who placed more than "
            "three orders in the last 30 days."
        ),
    },
    {
        "id": "dev_verification_001",
        "domain": "developer",
        "task": "verification",
        "prompt": (
            "An API returns HTTP 500 intermittently. "
            "Explain how to verify the root cause instead of guessing."
        ),
    },
    {
        "id": "dev_devops_001",
        "domain": "developer",
        "task": "devops",
        "prompt": (
            "A Docker container repeatedly restarts in production. "
            "List the evidence you would inspect."
        ),
    },
    {
        "id": "venture_market_001",
        "domain": "venture",
        "task": "market_analysis",
        "prompt": (
            "A startup operates in a large growing market. "
            "What evidence should be collected before judging the opportunity?"
        ),
    },
    {
        "id": "venture_traction_001",
        "domain": "venture",
        "task": "traction",
        "prompt": (
            "A startup reports 30 percent month-over-month revenue growth. "
            "What additional metrics should be verified?"
        ),
    },
    {
        "id": "venture_unit_economics_001",
        "domain": "venture",
        "task": "unit_economics",
        "prompt": (
            "A startup has LTV of 12000 and CAC of 3000. "
            "Calculate LTV:CAC and explain what must be verified."
        ),
    },
    {
        "id": "venture_risk_001",
        "domain": "venture",
        "task": "risk_analysis",
        "prompt": (
            "A startup has rapidly growing revenue but declining retention. "
            "What risks should be investigated?"
        ),
    },
    {
        "id": "venture_evidence_001",
        "domain": "venture",
        "task": "evidence_verification",
        "prompt": (
            "A founder claims that the company is growing 20 percent "
            "every month. Explain how VITO should verify the claim."
        ),
    },
]


def resolve_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_model(checkpoint, tokenizer_path, device):
    config = VitoConfig.from_pretrained(
        checkpoint
    )

    model = VitoForCausalLM(config)

    state_dict = load_file(
        str(checkpoint / "model.safetensors"),
        device="cpu",
    )

    model.load_state_dict(
        state_dict,
        strict=False,
    )

    model.tie_weights()
    model.to(device)
    model.eval()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path)
    )

    return model, tokenizer


@torch.no_grad()
def generate(
    model,
    tokenizer,
    device,
    prompt,
    max_new_tokens=80,
    temperature=0.7,
    top_k=50,
):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)

    for _ in range(max_new_tokens):
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
            temperature,
            1e-5,
        )

        top_k = min(
            top_k,
            logits.size(-1),
        )

        values, indices = torch.topk(
            logits,
            top_k,
        )

        filtered = torch.full_like(
            logits,
            float("-inf"),
        )

        filtered.scatter_(
            1,
            indices,
            values,
        )

        probabilities = torch.softmax(
            filtered,
            dim=-1,
        )

        next_token = torch.multinomial(
            probabilities,
            1,
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1,
        )

    return tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER,
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

    args = parser.parse_args()

    device = resolve_device()

    print("=" * 64)
    print("VITO-BENCH v0.1")
    print("=" * 64)
    print("Checkpoint:", args.checkpoint)
    print("Device:", device)
    print("Prompts:", len(PROMPTS))
    print()

    model, tokenizer = load_model(
        args.checkpoint,
        args.tokenizer,
        device,
    )

    results = []

    for index, item in enumerate(PROMPTS, start=1):
        print(
            f"[{index}/{len(PROMPTS)}] "
            f"{item['id']}"
        )

        output = generate(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=item["prompt"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        result = {
            **item,
            "output": output,
        }

        results.append(result)

        print(output)
        print("-" * 64)

    report = {
        "benchmark": "VITO-Bench",
        "version": "0.1.0",
        "model": {
            "checkpoint": str(
                args.checkpoint
            ),
            "parameters": sum(
                p.numel()
                for p in model.parameters()
            ),
        },
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        },
        "results": results,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 64)
    print("BENCHMARK COMPLETE")
    print("=" * 64)
    print("Results:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
