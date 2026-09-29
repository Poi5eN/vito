import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import VitoConfig, VitoForCausalLM


def make_model():
    config = VitoConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=32,
    )

    return VitoForCausalLM(config)


def test_save_and_load(tmp_path):
    torch.manual_seed(42)

    model = make_model()
    model.eval()

    input_ids = torch.randint(
        0,
        128,
        (1, 8),
    )

    with torch.no_grad():
        original = model(
            input_ids=input_ids
        ).logits

    save_dir = tmp_path / "vito-test"

    model.save_pretrained(
        save_dir,
        safe_serialization=True,
    )

    restored = VitoForCausalLM.from_pretrained(
        save_dir
    )

    restored.eval()

    with torch.no_grad():
        loaded = restored(
            input_ids=input_ids
        ).logits

    assert torch.allclose(
        original,
        loaded,
        atol=1e-5,
        rtol=1e-5,
    )


def test_config_round_trip(tmp_path):
    model = make_model()

    save_dir = tmp_path / "vito-config"

    model.save_pretrained(
        save_dir,
        safe_serialization=True,
    )

    restored = VitoForCausalLM.from_pretrained(
        save_dir
    )

    assert (
        restored.config.vocab_size
        == model.config.vocab_size
    )

    assert (
        restored.config.hidden_size
        == model.config.hidden_size
    )

    assert (
        restored.config.num_hidden_layers
        == model.config.num_hidden_layers
    )
