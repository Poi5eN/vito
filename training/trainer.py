from __future__ import annotations

import contextlib
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

        # -----------------------------------------------------------
        # Mixed precision
        # -----------------------------------------------------------
        #
        # FP16 is enabled only on CUDA devices such as the Kaggle T4.
        #
        # MPS and CPU continue using normal FP32 training.
        #
        self.use_amp = self.device.type == "cuda"

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        print(
            "Mixed precision: "
            f"{'FP16' if self.use_amp else 'disabled'}"
        )

        # -----------------------------------------------------------
        # Optimizer
        # -----------------------------------------------------------

        self.optimizer = create_optimizer(
            self.model,
            self.config,
        )

        # -----------------------------------------------------------
        # Learning-rate scheduler
        # -----------------------------------------------------------

        self.scheduler = create_scheduler(
            self.optimizer,
            self.config,
        )

        # -----------------------------------------------------------
        # Checkpoint manager
        # -----------------------------------------------------------

        self.checkpoints = VitoCheckpointManager(
            self.config.output_dir
        )

        self.global_step = 0

        self.best_validation_loss = float("inf")

        # -----------------------------------------------------------
        # Reproducibility
        # -----------------------------------------------------------

        self._set_seed(
            self.config.seed
        )

        # -----------------------------------------------------------
        # Resume training
        # -----------------------------------------------------------

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
                self.best_validation_loss = (
                    validation_loss
                )

            print(
                f"Resumed from step "
                f"{self.global_step} "
                f"with loss "
                f"{state['loss']:.6f}"
            )

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    def _resolve_device(self):

        if self.config.device != "auto":
            return torch.device(
                self.config.device
            )

        if torch.cuda.is_available():
            return torch.device("cuda")

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")

        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Random seed
    # ------------------------------------------------------------------

    def _set_seed(self, seed: int):

        random.seed(seed)

        np.random.seed(seed)

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Autocast context
    # ------------------------------------------------------------------

    def _autocast_context(self):

        if self.use_amp:

            return torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            )

        return contextlib.nullcontext()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

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
            ].to(
                self.device,
                non_blocking=self.use_amp,
            )

            labels = batch[
                "labels"
            ].to(
                self.device,
                non_blocking=self.use_amp,
            )

            # -------------------------------------------------------
            # FP16 autocast during validation
            # -------------------------------------------------------

            with self._autocast_context():

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

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

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
            ].to(
                self.device,
                non_blocking=self.use_amp,
            )

            labels = batch[
                "labels"
            ].to(
                self.device,
                non_blocking=self.use_amp,
            )

            # -------------------------------------------------------
            # Clear gradients
            # -------------------------------------------------------

            self.optimizer.zero_grad(
                set_to_none=True
            )

            # -------------------------------------------------------
            # Forward pass
            # -------------------------------------------------------
            #
            # On CUDA/T4 this executes supported operations using
            # FP16 where appropriate.
            #
            # On MPS/CPU this becomes a normal FP32 forward pass.
            #
            # -------------------------------------------------------

            with self._autocast_context():

                outputs = self.model(
                    input_ids=input_ids,
                    labels=labels,
                )

                loss = outputs.loss

            if loss is None:
                raise RuntimeError(
                    "Model returned no training loss."
                )

            # -------------------------------------------------------
            # Backward pass
            # -------------------------------------------------------
            #
            # GradScaler prevents FP16 gradient underflow.
            #
            # -------------------------------------------------------

            self.scaler.scale(
                loss
            ).backward()

            # -------------------------------------------------------
            # Unscale gradients BEFORE clipping
            # -------------------------------------------------------
            #
            # This is important.
            #
            # clip_grad_norm_ must operate on the real gradients,
            # not the scaled gradients.
            #
            # -------------------------------------------------------

            self.scaler.unscale_(
                self.optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip_norm,
            )

            # -------------------------------------------------------
            # Optimizer step
            # -------------------------------------------------------

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

            # -------------------------------------------------------
            # Learning-rate scheduler
            # -------------------------------------------------------

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