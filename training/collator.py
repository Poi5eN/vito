from __future__ import annotations

from typing import Sequence

import torch

from training.dataset import TrainingExample


class VitoDataCollator:
    """
    Converts individual TrainingExample objects into a batch.
    """

    def __call__(
        self,
        examples: Sequence[TrainingExample],
    ) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")

        input_ids = torch.stack(
            [example.input_ids for example in examples]
        )

        labels = torch.stack(
            [example.labels for example in examples]
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
        }
