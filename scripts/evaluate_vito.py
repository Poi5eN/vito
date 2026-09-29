from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import load_file

from model import VitoForCausalLM, VitoConfig
from training.dataset import VitoLanguageModelDataset
from training.collator import VitoDataCollator


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "vito-v0.2"
    / "checkpoint-500"
)

TOKENIZER_FILE = (
    PROJECT_ROOT
    / "tokenizer"
    / "vito-bpe-v0.2.json"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vito_corpus_v0.2"
    / "validation.txt"
)

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vito_corpus_v0.2"
    / "test.txt"
)

SEQUENCE_LENGTH = 128
BATCH_SIZE = 4


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_model(
    checkpoint_dir: Path,
    device: torch.device,
) -> VitoForCausalLM:

    config = VitoConfig.from_pretrained(
        checkpoint_dir
    )

    model = VitoForCausalLM(config)

    model_path = checkpoint_dir / "model.safetensors"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}"
        )

    state_dict = load_file(
        str(model_path),
        device="cpu",
    )

    model.load_state_dict(
        state_dict,
        strict=False,
    )

    # Intentional tied input/output embeddings.
    model.tie_weights()

    model.to(device)
    model.eval()

    return model


def evaluate(
    model: VitoForCausalLM,
    dataloader: DataLoader,
    device: torch.device,
) -> float:

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc="Evaluating",
        ):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            total_batches += 1

    if total_batches == 0:
        raise RuntimeError(
            "Evaluation dataset produced zero batches."
        )

    return total_loss / total_batches


def evaluate_split(
    model: VitoForCausalLM,
    file_path: Path,
    tokenizer_file: Path,
    device: torch.device,
    split_name: str,
) -> tuple[float, float, int, int]:

    if not file_path.exists():
        raise FileNotFoundError(
            f"{split_name} file not found: {file_path}"
        )

    dataset = VitoLanguageModelDataset(
        text_file=file_path,
        tokenizer_file=tokenizer_file,
        sequence_length=SEQUENCE_LENGTH,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=VitoDataCollator(),
    )

    print()
    print(f"{split_name} examples:", len(dataset))
    print(f"{split_name} tokens:", dataset.token_count())

    loss = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    perplexity = math.exp(loss)

    return (
        loss,
        perplexity,
        len(dataset),
        dataset.token_count(),
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Evaluate a VITO checkpoint."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    args = parser.parse_args()

    checkpoint_dir = args.checkpoint

    device = resolve_device()

    print("=" * 64)
    print("VITO v0.2 EVALUATION")
    print("=" * 64)

    print("Checkpoint:", checkpoint_dir)
    print("Tokenizer:", TOKENIZER_FILE)
    print("Validation:", VALIDATION_FILE)
    print("Test:", TEST_FILE)
    print("Sequence length:", SEQUENCE_LENGTH)
    print("Batch size:", BATCH_SIZE)
    print("Device:", device)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_dir}"
        )

    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_FILE}"
        )

    model = load_model(
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Parameters:", parameter_count)

    validation_loss, validation_ppl, _, _ = evaluate_split(
        model=model,
        file_path=VALIDATION_FILE,
        tokenizer_file=TOKENIZER_FILE,
        device=device,
        split_name="Validation",
    )

    test_loss, test_ppl, _, _ = evaluate_split(
        model=model,
        file_path=TEST_FILE,
        tokenizer_file=TOKENIZER_FILE,
        device=device,
        split_name="Test",
    )

    print()
    print("=" * 64)
    print("VITO v0.2 RESULTS")
    print("=" * 64)

    print(
        f"Validation loss:       {validation_loss:.6f}"
    )
    print(
        f"Validation perplexity:  {validation_ppl:.4f}"
    )

    print(
        f"Test loss:              {test_loss:.6f}"
    )
    print(
        f"Test perplexity:        {test_ppl:.4f}"
    )

    print("=" * 64)


if __name__ == "__main__":
    main()