from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

from model.configuration_vito import VitoConfig
from model.modeling_vito import VitoForCausalLM


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINT = ROOT / "experiments/vito-dev/checkpoint-100"
DEFAULT_TOKENIZER = ROOT / "tokenizer/vito-bpe-v0.1.json"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_vito(
    checkpoint_path=DEFAULT_CHECKPOINT,
    tokenizer_path=DEFAULT_TOKENIZER,
):
    device = get_device()

    config = VitoConfig.from_pretrained(
        checkpoint_path
    )

    model = VitoForCausalLM(config)

    state_dict = load_file(
        str(checkpoint_path / "model.safetensors"),
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

    return model, tokenizer, device


@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
):
    model, tokenizer, device = load_vito()

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)

    for _ in range(max_new_tokens):
        if input_ids.shape[1] > model.config.max_position_embeddings:
            input_ids = input_ids[
                :, -model.config.max_position_embeddings:
            ]

        outputs = model(
            input_ids=input_ids,
        )

        logits = outputs.logits[:, -1, :]

        logits = logits / max(temperature, 1e-5)

        if top_k is not None:
            top_k = min(top_k, logits.size(-1))

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

    return tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True,
    )