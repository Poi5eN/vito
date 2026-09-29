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


def test_forward_shape():
    model = make_model()

    input_ids = torch.randint(
        0,
        128,
        (2, 16),
    )

    outputs = model(input_ids)

    assert outputs.logits.shape == (
        2,
        16,
        128,
    )


def test_loss_and_backward():
    model = make_model()

    input_ids = torch.randint(
        0,
        128,
        (2, 16),
    )

    outputs = model(
        input_ids,
        labels=input_ids,
    )

    assert outputs.loss is not None

    outputs.loss.backward()

    assert (
        model.vito.token_embeddings.weight.grad
        is not None
    )


def test_weight_tying():
    model = make_model()

    input_weight = (
        model.get_input_embeddings().weight
    )

    output_weight = (
        model.get_output_embeddings().weight
    )

    assert (
        input_weight.data_ptr()
        == output_weight.data_ptr()
    )


def test_attention_is_causal():
    model = make_model()

    input_ids = torch.randint(
        0,
        128,
        (1, 8),
    )

    outputs = model(
        input_ids,
        output_attentions=True,
    )

    attention = outputs.attentions[0]

    upper_triangle = torch.triu(
        attention,
        diagonal=1,
    )

    assert torch.allclose(
        upper_triangle,
        torch.zeros_like(upper_triangle),
        atol=1e-5,
    )
