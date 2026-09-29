from pathlib import Path

import torch

from training.dataset import VitoLanguageModelDataset


ROOT = Path(__file__).resolve().parents[1]


def test_training_dataset_alignment():
    dataset = VitoLanguageModelDataset(
        text_file=ROOT / "data/processed/vito_corpus_v0.1/train.txt",
        tokenizer_file=ROOT / "tokenizer/vito-bpe-v0.1.json",
        sequence_length=32,
    )

    example = dataset[0]

    assert example.input_ids.dtype == torch.long
    assert example.labels.dtype == torch.long

    assert example.input_ids.shape == (32,)
    assert example.labels.shape == (32,)

    # Labels must be shifted one token to the right.
    combined = torch.cat(
        [
            example.input_ids,
            example.labels[-1:].clone(),
        ]
    )

    assert torch.equal(
        example.labels,
        combined[1:],
    )


def test_dataset_has_examples():
    dataset = VitoLanguageModelDataset(
        text_file=ROOT / "data/processed/vito_corpus_v0.1/train.txt",
        tokenizer_file=ROOT / "tokenizer/vito-bpe-v0.1.json",
        sequence_length=32,
    )

    assert len(dataset) > 0
    assert dataset.token_count() >= 33


def test_dataset_examples_are_fixed_length():
    dataset = VitoLanguageModelDataset(
        text_file=ROOT / "data/processed/vito_corpus_v0.1/train.txt",
        tokenizer_file=ROOT / "tokenizer/vito-bpe-v0.1.json",
        sequence_length=32,
    )

    for example in dataset.iter_examples():
        assert example.input_ids.shape == (32,)
        assert example.labels.shape == (32,)
