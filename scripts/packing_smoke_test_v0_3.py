from __future__ import annotations

import json
from pathlib import Path

import torch

from training.collator import VitoDataCollator
from training.packed_dataset import VitoPackedDataset


ROOT = Path(__file__).resolve().parents[1]
PACKED = ROOT / "data/processed/vito_packed_v0.3"


def main():
    print("=" * 64)
    print("VITO v0.3 PACKING SMOKE TEST")
    print("=" * 64)

    expected_length = 256

    datasets = {}

    for split in ("train", "validation", "test"):
        path = PACKED / f"{split}.jsonl"

        dataset = VitoPackedDataset(path)
        datasets[split] = dataset

        print(f"{split}: {len(dataset):,} packed sequences")

        example = dataset[0]

        assert example.input_ids.shape == (expected_length,)
        assert example.labels.shape == (expected_length,)

        assert example.input_ids.dtype == torch.long
        assert example.labels.dtype == torch.long

        assert torch.equal(
            example.input_ids[1:],
            example.labels[:-1],
        )

    collator = VitoDataCollator()

    batch = collator(
        [datasets["train"][0], datasets["train"][1]]
    )

    assert batch["input_ids"].shape == (2, expected_length)
    assert batch["labels"].shape == (2, expected_length)

    manifest = PACKED / "manifest.json"

    with manifest.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    assert metadata["sequence_length"] == expected_length
    assert metadata["vocab_size"] == 12288

    print()
    print("Batch input shape:", tuple(batch["input_ids"].shape))
    print("Batch label shape:", tuple(batch["labels"].shape))
    print("Sequence length:", expected_length)
    print()
    print("PACKING SMOKE TEST: PASSED")


if __name__ == "__main__":
    main()
