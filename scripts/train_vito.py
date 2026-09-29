from pathlib import Path

from torch.utils.data import DataLoader

from model import VitoConfig, VitoForCausalLM
from training.collator import VitoDataCollator
from training.config import TrainingConfig
from training.dataset import VitoLanguageModelDataset
from training.trainer import VitoTrainer


ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_PATH = (
    ROOT / "tokenizer/vito-bpe-v0.2.json"
)

TRAIN_FILE = (
    ROOT
    / "data/processed/vito_corpus_v0.2/train.txt"
)

OUTPUT_DIR = (
    ROOT / "experiments/vito-v0.2"
)


def main():
    print("=" * 64)
    print("VITO v0.2 TRAINING")
    print("=" * 64)

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    vocab_size = tokenizer.get_vocab_size()

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model_config = VitoConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        tie_word_embeddings=True,
    )

    model = VitoForCausalLM(model_config)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = VitoLanguageModelDataset(
        text_file=TRAIN_FILE,
        tokenizer_file=TOKENIZER_PATH,
        sequence_length=128,
    )

    # ---------------------------------------------------------
    # Training configuration
    # ---------------------------------------------------------

    training_config = TrainingConfig(
        batch_size=4,
        learning_rate=1e-4,
        weight_decay=0.01,
        max_steps=500,
        log_every=10,
        save_every=50,
        grad_clip_norm=1.0,
        seed=42,
        device="auto",
        output_dir=OUTPUT_DIR,
        resume_from=None,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=VitoDataCollator(),
    )

    # ---------------------------------------------------------
    # Training summary
    # ---------------------------------------------------------

    print()
    print("Tokenizer:", TOKENIZER_PATH)
    print("Vocabulary:", vocab_size)
    print("Parameters:", parameter_count)
    print("Training file:", TRAIN_FILE)
    print("Training examples:", len(dataset))
    print("Sequence length:", 128)
    print("Batch size:", training_config.batch_size)
    print("Learning rate:", training_config.learning_rate)
    print("Weight decay:", training_config.weight_decay)
    print("Max steps:", training_config.max_steps)
    print("Warmup steps:", int(training_config.max_steps * 0.05))
    print("Checkpoint every:", training_config.save_every)
    print("Output:", OUTPUT_DIR)
    print()

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    trainer = VitoTrainer(
        model=model,
        dataloader=dataloader,
        config=training_config,
    )

    losses = trainer.train()

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    print()
    print("=" * 64)
    print("VITO v0.2 TRAINING COMPLETE")
    print("=" * 64)

    print("Steps executed:", len(losses))

    if losses:
        print(
            "Initial loss:",
            f"{losses[0]:.6f}",
        )
        print(
            "Final loss:",
            f"{losses[-1]:.6f}",
        )
        print(
            "Minimum loss:",
            f"{min(losses):.6f}",
        )

    print(
        "Final global step:",
        trainer.global_step,
    )

    print(
        "Output directory:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()