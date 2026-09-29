from .configuration_vito import VitoConfig
from .modeling_vito import VitoForCausalLM


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":
    config = VitoConfig()

    model = VitoForCausalLM(config)

    total = count_parameters(model)

    print(f"Total parameters: {total:,}")
    print(f"Total parameters (M): {total / 1_000_000:.3f}")
