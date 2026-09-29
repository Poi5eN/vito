from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int = 4

    learning_rate: float = 1e-4

    weight_decay: float = 0.01

    max_steps: int = 100

    log_every: int = 10

    save_every: int = 50

    eval_every: int = 500

    grad_clip_norm: float = 1.0

    seed: int = 42

    device: str = "auto"

    output_dir: str = "experiments/vito-dev"

    resume_from: str | None = None
