from __future__ import annotations

import torch

from training.config import TrainingConfig


def create_optimizer(
    model: torch.nn.Module,
    config: TrainingConfig,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
):
    warmup_steps = max(
        1,
        int(config.max_steps * 0.05),
    )

    decay_steps = max(
        1,
        config.max_steps - warmup_steps,
    )

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        progress = min(
            1.0,
            (step - warmup_steps) / decay_steps,
        )

        min_lr_ratio = 0.1

        cosine = 0.5 * (
            1.0
            + torch.cos(
                torch.tensor(
                    torch.pi * progress
                )
            ).item()
        )

        return min_lr_ratio + (
            1.0 - min_lr_ratio
        ) * cosine

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )