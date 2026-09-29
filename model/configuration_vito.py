from transformers import PretrainedConfig


class VitoConfig(PretrainedConfig):
    """
    Configuration for the VITO decoder-only Transformer.

    VITO v0.1:
    - Decoder-only
    - Learned positional embeddings
    - Pre-LayerNorm
    - Causal self-attention
    - GELU MLP
    - Tied token embeddings / LM head
    """

    model_type = "vito"

    def __init__(
        self,
        vocab_size=12288,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.0,
        initializer_range=0.02,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.initializer_range = initializer_range

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )

        self.head_dim = hidden_size // num_attention_heads
