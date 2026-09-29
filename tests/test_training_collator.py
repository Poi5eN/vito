from pathlib import Path

import torch

from training.collator import VitoDataCollator
from training.dataset import VitoLanguageModelDataset


ROOT = Path(__file__).resolve().parents[1]


def make_dataset():
    return VitoLanguageModelDataset(
        text_file=ROOT / "data/processed/vito_corpus_v0.1/train.txt",
        tokenizer_file=ROOT / "tokenizer/vito-bpe-v0.1.json",
        sequence_length=32,
    )


def test_collator_shapes():
    dataset = make_dataset()
    collator = VitoDataCollator()

    examples = [
        dataset[0],
        dataset[1],
        dataset[2],
        dataset[3],
    ]

    batch = collator(examples)

    assert batch["input_ids"].shape == (4, 32)
    assert batch["labels"].shape == (4, 32)

    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long


def test_collator_preserves_examples():
    dataset = make_dataset()
    collator = VitoDataCollator()

    examples = [
        dataset[0],
        dataset[1],
    ]

    batch = collator(examples)

    assert torch.equal(
        batch["input_ids"][0],
        examples[0].input_ids,
    )

    assert torch.equal(
        batch["input_ids"][1],
        examples[1].input_ids,
    )

    assert torch.equal(
        batch["labels"][0],
        examples[0].labels,
    )

    assert torch.equal(
        batch["labels"][1],
        examples[1].labels,
    )


def test_collator_rejects_empty_batch():
    collator = VitoDataCollator()

    try:
        collator([])
    except ValueError:
        return

    raise AssertionError("Expected ValueError for empty batch")
