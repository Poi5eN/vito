from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class PackedTrainingExample:
    input_ids: torch.Tensor
    labels: torch.Tensor


class VitoPackedDataset(Dataset):
    """
    Dataset for pre-packed VITO v0.3 causal-LM sequences.

    Each JSONL row contains:
        {
            "input_ids": [...],
            "labels": [...]
        }
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self._offsets: list[int] = []

        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()

                if not line:
                    break

                if line.strip():
                    self._offsets.append(offset)

        if not self._offsets:
            raise ValueError(f"Packed dataset is empty: {self.path}")

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> PackedTrainingExample:
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset "
                f"of size {len(self)}"
            )

        with self.path.open("rb") as handle:
            handle.seek(self._offsets[index])
            line = handle.readline()

        record = json.loads(line)

        input_ids = torch.tensor(
            record["input_ids"],
            dtype=torch.long,
        )

        labels = torch.tensor(
            record["labels"],
            dtype=torch.long,
        )

        return PackedTrainingExample(
            input_ids=input_ids,
            labels=labels,
        )
