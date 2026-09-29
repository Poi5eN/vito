from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from model import VitoConfig, VitoForCausalLM
from training.collator import VitoDataCollator
from training.config import TrainingConfig
from training.packed_dataset import VitoPackedDataset
from training.trainer import VitoTrainer


ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_PATH = ROOT / "tokenizer/vito-bpe-v0.3.json"
PACKED_DIR = ROOT / "data/processed/vito_packed_v0.3"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train VITO v0.3."
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/vito-v0.3",
    )

    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=500,
    )

    return parser.parse_args()


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):

            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                labels=labels,
            )

            loss = outputs.loss

            if loss is None:
                raise RuntimeError(
                    "Model returned no loss during evaluation."
                )

            total_loss += float(
                loss.detach().cpu()
            )

            total_batches += 1

    if total_batches == 0:
        raise RuntimeError(
            "Evaluation dataloader produced zero batches."
        )

    mean_loss = total_loss / total_batches

    try:
        perplexity = math.exp(mean_loss)
    except OverflowError:
        perplexity = float("inf")

    model.train()

    return mean_loss, perplexity


def build_model(vocab_size: int):
    config = VitoConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=256,
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        tie_word_embeddings=True,
    )

    return VitoForCausalLM(config)


def main():

    args = parse_args()

    output_dir = ROOT / args.output_dir

    device_name = detect_device()
    device = torch.device(device_name)

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    vocab_size = tokenizer.get_vocab_size()

    print("=" * 72)
    print("VITO v0.3 TRAINING")
    print("=" * 72)

    print(f"Tokenizer: {TOKENIZER_PATH}")
    print(f"Vocabulary: {vocab_size}")
    print(f"Device: {device}")
    print()

    model = build_model(vocab_size)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: {parameter_count:,}"
    )

    train_dataset = VitoPackedDataset(
        PACKED_DIR / "train.jsonl"
    )

    validation_dataset = VitoPackedDataset(
        PACKED_DIR / "validation.jsonl"
    )

    test_dataset = VitoPackedDataset(
        PACKED_DIR / "test.jsonl"
    )

    print(
        f"Training sequences:   {len(train_dataset):,}"
    )

    print(
        f"Validation sequences: {len(validation_dataset):,}"
    )

    print(
        f"Test sequences:       {len(test_dataset):,}"
    )

    print(
        "Sequence length:      256"
    )

    print()

    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        eval_every=args.eval_every,
        grad_clip_norm=1.0,
        seed=42,
        device=device_name,
        output_dir=str(output_dir),
        resume_from=args.resume_from,
    )

    collator = VitoDataCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    model.to(device)

    print("Batch size:", config.batch_size)
    print("Learning rate:", config.learning_rate)
    print("Weight decay:", config.weight_decay)
    print("Max steps:", config.max_steps)
    print("Save every:", config.save_every)
    print("Log every:", config.log_every)
    print("Output:", output_dir)

    if args.resume_from:
        print(
            "Resume from:",
            args.resume_from,
        )

    print()

    # ---------------------------------------------------------------
    # Initial evaluation
    # ---------------------------------------------------------------

    print("=" * 72)
    print("INITIAL VALIDATION")
    print("=" * 72)

    initial_val_loss, initial_val_ppl = (
        evaluate_model(
            model,
            validation_loader,
            device,
        )
    )

    print(
        f"Validation loss: {initial_val_loss:.6f}"
    )

    print(
        f"Validation perplexity: {initial_val_ppl:.4f}"
    )

    print()

    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------

    trainer = VitoTrainer(
        model=model,
        dataloader=train_loader,
        validation_dataloader=validation_loader,
        config=config,
)

    losses = trainer.train()

    # ---------------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------------

    final_val_loss, final_val_ppl = (
        evaluate_model(
            model,
            validation_loader,
            device,
        )
    )

    # ---------------------------------------------------------------
    # Final test
    # ---------------------------------------------------------------

    final_test_loss, final_test_ppl = (
        evaluate_model(
            model,
            test_loader,
            device,
        )
    )

    print()
    print("=" * 72)
    print("VITO v0.3 TRAINING COMPLETE")
    print("=" * 72)

    print(
        f"Steps executed: {len(losses)}"
    )

    if losses:
        print(
            f"Initial train loss: {losses[0]:.6f}"
        )

        print(
            f"Final train loss:   {losses[-1]:.6f}"
        )

        print(
            f"Minimum train loss: {min(losses):.6f}"
        )

    print()

    print(
        f"Initial validation loss: "
        f"{initial_val_loss:.6f}"
    )

    print(
        f"Final validation loss: "
        f"{final_val_loss:.6f}"
    )

    print(
        f"Final validation perplexity: "
        f"{final_val_ppl:.4f}"
    )

    print()

    print(
        f"Final test loss: "
        f"{final_test_loss:.6f}"
    )

    print(
        f"Final test perplexity: "
        f"{final_test_ppl:.4f}"
    )

    print()

    print(
        "Final global step:",
        trainer.global_step,
    )

    print(
        "Output directory:",
        output_dir,
    )


if __name__ == "__main__":
    main()
