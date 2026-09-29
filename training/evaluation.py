from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from training.metrics import perplexity


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:

    model.eval()

    total_loss = 0.0
    batches = 0

    for batch in dataloader:

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            labels=labels,
        )

        loss = outputs.loss

        total_loss += loss.item()
        batches += 1

    mean_loss = (
        total_loss / batches
        if batches
        else float("inf")
    )

    model.train()

    return {
        "loss": mean_loss,
        "perplexity": perplexity(mean_loss),
    }
