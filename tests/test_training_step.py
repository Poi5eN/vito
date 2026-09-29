from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import VitoConfig, VitoForCausalLM
from training.collator import VitoDataCollator
from training.dataset import VitoLanguageModelDataset


ROOT = Path(__file__).resolve().parents[1]


def make_dataset():
    return VitoLanguageModelDataset(
        text_file=ROOT / "data/processed/vito_corpus_v0.1/train.txt",
        tokenizer_file=ROOT / "tokenizer/vito-bpe-v0.1.json",
        sequence_length=32,
    )


def make_model():
    return VitoForCausalLM(
        VitoConfig(
            vocab_size=1060,
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=2048,
            max_position_embeddings=512,
        )
    )


def test_single_training_step():
    torch.manual_seed(42)

    dataset = make_dataset()

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=VitoDataCollator(),
    )

    batch = next(iter(loader))

    model = make_model()
    model.train()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model.to(device)

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(
        input_ids=input_ids,
        labels=labels,
    )

    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)

    loss = outputs.loss

    model.zero_grad(set_to_none=True)
    loss.backward()

    gradients_found = 0

    for parameter in model.parameters():
        if parameter.grad is not None:
            gradients_found += 1

    assert gradients_found > 0

    print(f"Training device: {device}")
    print(f"Loss: {loss.item():.6f}")
    print(f"Parameters with gradients: {gradients_found}")

    assert outputs.logits.shape == (
        4,
        32,
        1060,
    )
