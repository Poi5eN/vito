from __future__ import annotations

import math
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
        validation_dataloader: DataLoader | None = None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.validation_dataloader = validation_dataloader
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

        self.best_validation_loss = float("inf")

        self._set_seed(
            self.config.seed
        )

        if getattr(
            self.config,
            "resume_from",
            None,
        ):

            state = self.checkpoints.load(
                checkpoint_dir=self.config.resume_from,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )

            self.global_step = state["step"]

            validation_loss = state.get(
                "validation_loss"
            )

            if validation_loss is not None:
                self.best_validation_loss = validation_loss

            print(
                f"Resumed from step "
                f"{self.global_step} "
                f"with loss "
                f"{state['loss']:.6f}"
            )

    def _resolve_device(self):

        if self.config.device != "auto":
            return torch.device(
                self.config.device
            )

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    def _set_seed(self, seed: int):

        random.seed(seed)

        np.random.seed(seed)

        torch.manual_seed(seed)

    @torch.no_grad()
    def evaluate(self):

        if self.validation_dataloader is None:
            return None, None

        self.model.eval()

        total_loss = 0.0
        total_batches = 0

        for batch in self.validation_dataloader:

            input_ids = batch[
                "input_ids"
            ].to(self.device)

            labels = batch[
                "labels"
            ].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                labels=labels,
            )

            loss = outputs.loss

            if loss is None:
                raise RuntimeError(
                    "Model returned no loss during validation."
                )

            total_loss += float(
                loss.detach().cpu()
            )

            total_batches += 1

        if total_batches == 0:
            raise RuntimeError(
                "Validation dataloader "
                "produced zero batches."
            )

        validation_loss = (
            total_loss / total_batches
        )

        try:
            validation_perplexity = math.exp(
                validation_loss
            )
        except OverflowError:
            validation_perplexity = float("inf")

        self.model.train()

        return (
            validation_loss,
            validation_perplexity,
        )

    def train(self):

        self.model.train()

        data_iterator = iter(
            self.dataloader
        )

        losses = []

        while (
            self.global_step
            < self.config.max_steps
        ):

            try:

                batch = next(
                    data_iterator
                )

            except StopIteration:

                data_iterator = iter(
                    self.dataloader
                )

                batch = next(
                    data_iterator
                )

            input_ids = batch[
                "input_ids"
            ].to(self.device)

            labels = batch[
                "labels"
            ].to(self.device)

            self.optimizer.zero_grad(
                set_to_none=True
            )

            outputs = self.model(
                input_ids=input_ids,
                labels=labels,
            )

            loss = outputs.loss

            if loss is None:
                raise RuntimeError(
                    "Model returned no training loss."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip_norm,
            )

            self.optimizer.step()

            self.scheduler.step()

            self.global_step += 1

            loss_value = loss.item()

            losses.append(
                loss_value
            )

            # -------------------------------------------------------
            # Training log
            # -------------------------------------------------------

            if (
                self.global_step
                % self.config.log_every
                == 0
            ):

                print(
                    f"step={self.global_step} "
                    f"loss={loss_value:.6f} "
                    f"lr="
                    f"{self.scheduler.get_last_lr()[0]:.8f}"
                )

            # -------------------------------------------------------
            # Validation
            # -------------------------------------------------------

            should_evaluate = (
                self.validation_dataloader
                is not None
                and (
                    self.global_step
                    % self.config.eval_every
                    == 0
                )
            )

            validation_loss = None
            validation_perplexity = None
            is_best = False

            if should_evaluate:

                (
                    validation_loss,
                    validation_perplexity,
                ) = self.evaluate()

                is_best = (
                    validation_loss
                    < self.best_validation_loss
                )

                if is_best:
                    self.best_validation_loss = (
                        validation_loss
                    )

                print(
                    f"step={self.global_step} "
                    f"val_loss="
                    f"{validation_loss:.6f} "
                    f"val_ppl="
                    f"{validation_perplexity:.4f}"
                )

                if is_best:
                    print(
                        "NEW BEST VALIDATION CHECKPOINT"
                    )

            # -------------------------------------------------------
            # Checkpoint
            # -------------------------------------------------------

            should_save = (
                self.global_step
                % self.config.save_every
                == 0
            )

            if should_save:

                checkpoint = (
                    self.checkpoints.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        step=self.global_step,
                        loss=loss_value,
                        validation_loss=validation_loss,
                        validation_perplexity=(
                            validation_perplexity
                        ),
                        is_best=is_best,
                    )
                )

                print(
                    f"checkpoint_saved="
                    f"{checkpoint}"
                )

        return losses
