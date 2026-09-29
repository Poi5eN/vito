from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset

from tokenizers import Tokenizer


@dataclass
class TrainingExample:
    input_ids: torch.Tensor
    labels: torch.Tensor


class VitoLanguageModelDataset(Dataset):
    """
    Converts a plain-text corpus into fixed-length next-token
    prediction examples.

    For tokens:

        [A, B, C, D, E]

    the dataset produces:

        input_ids = [A, B, C, D]
        labels    = [B, C, D, E]
    """

    def __init__(
        self,
        text_file: str | Path,
        tokenizer_file: str | Path,
        sequence_length: int,
    ) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")

        self.text_file = Path(text_file)
        self.tokenizer_file = Path(tokenizer_file)
        self.sequence_length = sequence_length

        if not self.text_file.exists():
            raise FileNotFoundError(
                f"Text file not found: {self.text_file}"
            )

        if not self.tokenizer_file.exists():
            raise FileNotFoundError(
                f"Tokenizer file not found: {self.tokenizer_file}"
            )

        self.tokenizer = Tokenizer.from_file(
            str(self.tokenizer_file)
        )

        text = self.text_file.read_text(encoding="utf-8")

        if not text.strip():
            raise ValueError("Training text is empty")

        encoded = self.tokenizer.encode(text)
        self.token_ids = encoded.ids

        if len(self.token_ids) < self.sequence_length + 1:
            raise ValueError(
                "Corpus does not contain enough tokens for "
                f"sequence_length={self.sequence_length}. "
                f"Found {len(self.token_ids)} tokens."
            )

        self.num_examples = (
            len(self.token_ids) - 1
        ) // self.sequence_length

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> TrainingExample:
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset of "
                f"size {len(self)}"
            )

        start = index * self.sequence_length
        end = start + self.sequence_length + 1

        tokens = self.token_ids[start:end]

        input_ids = torch.tensor(
            tokens[:-1],
            dtype=torch.long,
        )

        labels = torch.tensor(
            tokens[1:],
            dtype=torch.long,
        )

        return TrainingExample(
            input_ids=input_ids,
            labels=labels,
        )

    def token_count(self) -> int:
        return len(self.token_ids)

    def iter_examples(self) -> Iterator[TrainingExample]:
        for index in range(len(self)):
            yield self[index]