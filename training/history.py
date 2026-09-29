from __future__ import annotations

import json
from pathlib import Path


class TrainingHistory:

    def __init__(self):
        self.records: list[dict] = []

    def add(
        self,
        step: int,
        loss: float,
        learning_rate: float,
    ):
        self.records.append(
            {
                "step": step,
                "loss": float(loss),
                "learning_rate": float(
                    learning_rate
                ),
            }
        )

    def save(
        self,
        path: str | Path,
    ):
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                self.records,
                indent=2,
            ),
            encoding="utf-8",
        )
