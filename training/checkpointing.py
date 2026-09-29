from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file


class VitoCheckpointManager:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        model,
        optimizer,
        scheduler,
        step: int,
        loss: float,
        validation_loss: float | None = None,
        validation_perplexity: float | None = None,
        is_best: bool = False,
    ) -> Path:

        checkpoint_dir = (
            self.output_dir / f"checkpoint-{step}"
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model.save_pretrained(
            checkpoint_dir,
            safe_serialization=True,
        )

        torch.save(
            optimizer.state_dict(),
            checkpoint_dir / "optimizer.pt",
        )

        torch.save(
            scheduler.state_dict(),
            checkpoint_dir / "scheduler.pt",
        )

        state = {
            "step": step,
            "loss": loss,
            "validation_loss": validation_loss,
            "validation_perplexity": validation_perplexity,
            "is_best": is_best,
        }

        torch.save(
            state,
            checkpoint_dir / "trainer_state.pt",
        )

        if is_best:
            best_file = (
                self.output_dir / "best_checkpoint.txt"
            )

            best_file.write_text(
                str(checkpoint_dir),
                encoding="utf-8",
            )

        return checkpoint_dir

    def load(
        self,
        checkpoint_dir: str | Path,
        model,
        optimizer,
        scheduler,
    ) -> dict:

        checkpoint_dir = Path(checkpoint_dir)

        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_dir}"
            )

        model_state = checkpoint_dir / "model.safetensors"
        optimizer_state = checkpoint_dir / "optimizer.pt"
        scheduler_state = checkpoint_dir / "scheduler.pt"
        trainer_state = checkpoint_dir / "trainer_state.pt"

        required_files = [
            model_state,
            optimizer_state,
            scheduler_state,
            trainer_state,
        ]

        for file in required_files:
            if not file.exists():
                raise FileNotFoundError(
                    f"Missing checkpoint file: {file}"
                )

        state_dict = load_file(
            str(model_state)
        )

        model.load_state_dict(
            state_dict,
            strict=False,
        )

        # Re-establish tied lm_head relationship.
        model.tie_weights()

        optimizer.load_state_dict(
            torch.load(
                optimizer_state,
                map_location="cpu",
                weights_only=False,
            )
        )

        scheduler.load_state_dict(
            torch.load(
                scheduler_state,
                map_location="cpu",
                weights_only=False,
            )
        )

        state = torch.load(
            trainer_state,
            map_location="cpu",
            weights_only=False,
        )

        return state
