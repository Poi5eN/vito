import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .configuration_vito import VitoConfig


class VitoPreTrainedModel(PreTrainedModel):
    config_class = VitoConfig
    base_model_prefix = "vito"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0,
                std=self.config.initializer_range,
            )

            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class VitoSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        self.qkv = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=True,
        )

        self.out_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
        )

        self.attention_dropout = nn.Dropout(
            config.attention_dropout_prob
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):
        batch_size, seq_len, hidden_size = hidden_states.shape

        qkv = self.qkv(hidden_states)

        qkv = qkv.view(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )

        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        )

        scores = scores / math.sqrt(self.head_dim)

        # Causal mask:
        # token i may only attend to tokens <= i.
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=hidden_states.device,
            ),
            diagonal=1,
        )

        scores = scores.masked_fill(
            causal_mask,
            torch.finfo(scores.dtype).min,
        )

        # Optional padding mask.
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].bool()

            scores = scores.masked_fill(
                ~key_mask,
                torch.finfo(scores.dtype).min,
            )

        attention_weights = F.softmax(
            scores,
            dim=-1,
        )

        attention_weights = self.attention_dropout(
            attention_weights
        )

        context = torch.matmul(
            attention_weights,
            value,
        )

        context = context.transpose(1, 2).contiguous()

        context = context.view(
            batch_size,
            seq_len,
            hidden_size,
        )

        output = self.out_proj(context)

        if output_attentions:
            return output, attention_weights

        return output, None


class VitoMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc_in = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
        )

        self.fc_out = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
        )

        self.dropout = nn.Dropout(
            config.hidden_dropout_prob
        )

    def forward(self, hidden_states):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class VitoBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(
            config.hidden_size
        )

        self.attention = VitoSelfAttention(config)

        self.ln_2 = nn.LayerNorm(
            config.hidden_size
        )

        self.mlp = VitoMLP(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):
        # Pre-LN attention
        residual = hidden_states

        hidden_states = self.ln_1(hidden_states)

        attention_output, attention_weights = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = residual + attention_output

        # Pre-LN MLP
        residual = hidden_states

        hidden_states = self.ln_2(hidden_states)

        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, attention_weights


class VitoModel(VitoPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )

        self.dropout = nn.Dropout(
            config.hidden_dropout_prob
        )

        self.layers = nn.ModuleList(
            [
                VitoBlock(config)
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.token_embeddings

    def set_input_embeddings(self, value):
        self.token_embeddings = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "Specify either input_ids or inputs_embeds, not both."
            )

        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)

        batch_size, seq_len, _ = inputs_embeds.shape

        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"maximum context length "
                f"{self.config.max_position_embeddings}."
            )

        if position_ids is None:
            position_ids = torch.arange(
                seq_len,
                device=inputs_embeds.device,
            ).unsqueeze(0)

        position_embeddings = self.position_embeddings(
            position_ids
        )

        hidden_states = (
            inputs_embeds + position_embeddings
        )

        hidden_states = self.dropout(hidden_states)

        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        for layer in self.layers:
            hidden_states, attention_weights = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if output_attentions:
                all_attentions.append(attention_weights)

        hidden_states = self.final_layer_norm(
            hidden_states
        )

        if output_hidden_states:
            all_hidden_states[-1] = hidden_states

        return (
            hidden_states,
            tuple(all_hidden_states)
            if output_hidden_states
            else None,
            tuple(all_attentions)
            if output_attentions
            else None,
        )


class VitoForCausalLM(VitoPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.vito = VitoModel(config)

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        self.post_init()

        self.tie_weights()

    def get_input_embeddings(self):
        return self.vito.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.vito.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self, missing_keys=None, recompute_mapping=True):
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.vito.token_embeddings.weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if output_attentions is None:
            output_attentions = self.config.output_attentions

        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        if return_dict is None:
            return_dict = self.config.return_dict

        hidden_states, all_hidden_states, all_attentions = (
            self.vito(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        )

        logits = self.lm_head(hidden_states)

        loss = None

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (
                logits,
                all_hidden_states,
                all_attentions,
            )

            if loss is not None:
                output = (loss,) + output

            return output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
