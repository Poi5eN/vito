from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.checkpointing import VitoCheckpointManager
from training.config import TrainingConfig
from training.optimizer import (
    create_optimizer,
    create_scheduler,
)


class VitoTrainer:

    def __init__(
        self,
        model,
        dataloader: DataLoader,
        config: TrainingConfig,
    ):
        self.model = model
        self.dataloader = dataloader
        self.config = config

        self.device = self._resolve_device()

        self.model.to(self.device)

        self.optimizer = create_optimizer(
            self.model,
            self.config,
        )

        self.scheduler = create_scheduler(
            self.optimizer,
            self.config,
        )

        self.checkpoints = VitoCheckpointManager(
            self.config.output_dir
        )

        self.global_step = 0

        self._set_seed(self.config.seed)

        if getattr(self.config, "resume_from", None):
            state = self.checkpoints.load(
                checkpoint_dir=self.config.resume_from,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )

            self.global_step = state["step"]

            print(
                f"Resumed from step {self.global_step} "
                f"with loss {state['loss']:.6f}"
            )

    def _resolve_device(self):
        if self.config.device != "auto":
            return torch.device(self.config.device)

        if torch.backends.mps.is_available():
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def train(self):

        self.model.train()

        data_iterator = iter(self.dataloader)

        losses = []

        while self.global_step < self.config.max_steps:

            try:
                batch = next(data_iterator)

            except StopIteration:
                data_iterator = iter(self.dataloader)
                batch = next(data_iterator)

            input_ids = batch["input_ids"].to(
                self.device
            )

            labels = batch["labels"].to(
                self.device
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            outputs = self.model(
                input_ids=input_ids,
                labels=labels,
            )

            loss = outputs.loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip_norm,
            )

            self.optimizer.step()

            self.scheduler.step()

            self.global_step += 1

            loss_value = loss.item()

            losses.append(loss_value)

            if (
                self.global_step % self.config.log_every
                == 0
            ):
                print(
                    f"step={self.global_step} "
                    f"loss={loss_value:.6f} "
                    f"lr={self.scheduler.get_last_lr()[0]:.8f}"
                )

            if (
                self.global_step % self.config.save_every
                == 0
            ):
                self.checkpoints.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    step=self.global_step,
                    loss=loss_value,
                )

        return losses
