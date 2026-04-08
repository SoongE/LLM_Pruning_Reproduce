import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


class DecoderLM(nn.Module):
    def __init__(self, config_name, config, target_layer, max_length):
        super(DecoderLM, self).__init__()
        # config._attn_implementation_internal = 'flash_attention_2'
        config._attn_implementation_internal = 'sdpa'
        if target_layer == 0:
            raise ValueError(f'target_layer 0 is not supported.')

        self.config_name = config_name
        self.config = config
        self.target_layer = target_layer
        self.is_last_layer = self.target_layer == config.num_hidden_layers - 1

        self.decoder_layer, self.norm = self.create_layer()
        self.rotary_emb = self.embed_module(config)
        self.register_buffer('position_ids', torch.arange(0, max_length).unsqueeze(0))

    def create_layer(self):
        model = AutoModelForCausalLM.from_pretrained(self.config_name)
        layer = model.model.layers[self.target_layer]

        if self.is_last_layer:
            norm = model.norm
        else:
            norm = None
        return layer, norm

    def state_dict(self, *args):
        state_dict = {}
        prefix = f'model.layers.{self.target_layer}.'
        for key, value in self.decoder_layer.state_dict().items():
            new_key = prefix + key
            state_dict[new_key] = value

        if self.norm:
            state_dict['norm'] = self.norm
        return state_dict

    @property
    def embed_module(self):
        return LlamaRotaryEmbedding

    def forward(self, hidden_states):
        position_embeddings = self.rotary_emb(hidden_states, self.position_ids)

        hidden_states = self.decoder_layer(
            hidden_states,
            attention_mask=None,
            position_ids=self.position_ids,
            past_key_value=None,
            use_cache=False,
            cache_position=None,
            position_embeddings=position_embeddings,
        )

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        return hidden_states
