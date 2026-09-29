from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import VitoConfig, VitoForCausalLM
from training.collator import VitoDataCollator
from training.config import TrainingConfig
from training.dataset import VitoLanguageModelDataset
from training.trainer import VitoTrainer


ROOT = Path(__file__).resolve().parents[1]


def test_trainer_runs_multiple_steps(tmp_path):

    torch.manual_seed(42)

    model = VitoForCausalLM(
        VitoConfig(
            vocab_size=1060,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            max_position_embeddings=64,
        )
    )

    dataset = VitoLanguageModelDataset(
        text_file=(
            ROOT
            / "data/processed/vito_corpus_v0.1/train.txt"
        ),
        tokenizer_file=(
            ROOT
            / "tokenizer/vito-bpe-v0.1.json"
        ),
        sequence_length=32,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=VitoDataCollator(),
    )

    config = TrainingConfig(
        batch_size=2,
        learning_rate=1e-4,
        max_steps=5,
        log_every=1,
        save_every=100,
        output_dir=str(tmp_path),
    )

    trainer = VitoTrainer(
        model=model,
        dataloader=dataloader,
        config=config,
    )

    losses = trainer.train()

    assert len(losses) == 5
    assert trainer.global_step == 5

    assert all(
        torch.isfinite(
            torch.tensor(loss)
        )
        for loss in losses
    )
