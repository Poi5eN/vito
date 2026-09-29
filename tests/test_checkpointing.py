from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import VitoConfig, VitoForCausalLM
from training.collator import VitoDataCollator
from training.config import TrainingConfig
from training.dataset import VitoLanguageModelDataset
from training.trainer import VitoTrainer


ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_can_be_created_and_resumed(tmp_path):

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
        max_steps=2,
        log_every=1,
        save_every=2,
        output_dir=str(tmp_path),
    )

    trainer = VitoTrainer(
        model=model,
        dataloader=dataloader,
        config=config,
    )

    trainer.train()

    checkpoint = (
        tmp_path / "checkpoint-2"
    )

    assert (
        checkpoint / "model.safetensors"
    ).exists()

    assert (
        checkpoint / "optimizer.pt"
    ).exists()

    assert (
        checkpoint / "scheduler.pt"
    ).exists()

    assert (
        checkpoint / "trainer_state.pt"
    ).exists()

    resumed_model = VitoForCausalLM(
        VitoConfig(
            vocab_size=1060,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            max_position_embeddings=64,
        )
    )

    resumed_config = TrainingConfig(
        batch_size=2,
        max_steps=4,
        log_every=1,
        save_every=100,
        output_dir=str(tmp_path),
        resume_from=str(checkpoint),
)

    resumed_trainer = VitoTrainer(
        model=resumed_model,
        dataloader=dataloader,
        config=resumed_config,
    )

    assert resumed_trainer.global_step == 2

    losses = resumed_trainer.train()

    assert len(losses) == 2
    assert resumed_trainer.global_step == 4
