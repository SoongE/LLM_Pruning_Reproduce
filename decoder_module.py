import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding


class DecoderLM(nn.Module):
    def __init__(self, config_name, config, best_layer, layer_interval, num_layer, max_length):
        super(DecoderLM, self).__init__()
        # config._attn_implementation_internal = 'flash_attention_2'
        config._attn_implementation_internal = 'sdpa'
        if best_layer == 0:
            raise ValueError(f'best_layer 0 is not supported.')

        self.config_name = config_name
        self.config = config
        self.save_layer = best_layer
        self.target_layer = best_layer + layer_interval

        if num_layer == 1:
            self.decoder_layer, self.norm = self.create_layer(self.target_layer)
            self.last_decoder_layer = None
        elif num_layer == 2:
            assert self.target_layer + 1 == config.num_hidden_layers - 1
            self.decoder_layer, _ = self.create_layer(self.target_layer)
            self.last_decoder_layer, self.norm = self.create_layer(self.target_layer + 1)
        else:
            raise ValueError(f'num_layer: {num_layer} is not supported.')

        self.rotary_emb = self.get_embed_module(config_name, config)
        self.register_buffer('position_ids', torch.arange(0, max_length).unsqueeze(0))

    def create_layer(self, target_layer):
        model = AutoModelForCausalLM.from_pretrained(self.config_name)
        layer = model.model.layers[target_layer]

        is_last_layer = target_layer == self.config.num_hidden_layers - 1

        if is_last_layer:
            norm = model.model.norm
        else:
            norm = None
        return layer, norm

    def state_dict(self, *args):
        state_dict = {}
        prefix = f'model.layers.{self.save_layer}.'
        for key, value in self.decoder_layer.state_dict().items():
            new_key = prefix + key
            state_dict[new_key] = value

        if self.last_decoder_layer:
            prefix = f'model.layers.{self.save_layer + 1}.'
            for key, value in self.last_decoder_layer.state_dict().items():
                new_key = prefix + key
                state_dict[new_key] = value

        if self.norm:
            state_dict['norm'] = self.norm
        return state_dict

    def get_embed_module(self, config_name, config):
        config_name = config_name.lower()
        if 'llama' in config_name:
            return LlamaRotaryEmbedding(config)
        elif 'qwen' in config_name:
            return Qwen3RotaryEmbedding(config)
        else:
            raise ValueError(f'config_name: {config_name} is not supported.')

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
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        if self.last_decoder_layer:
            hidden_states = self.last_decoder_layer(
                hidden_states,
                attention_mask=None,
                position_ids=self.position_ids,
                past_key_value=None,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        return hidden_states


if __name__ == '__main__':
    config_name = 'meta-llama/Llama-3.2-3B'
    # config_name = 'Qwen/Qwen3-4B-Base'
    config = AutoConfig.from_pretrained(config_name)
    model = DecoderLM(config_name, config, 22, 4, 2, 1024)
    x = torch.rand(5, 1024, config.hidden_size)

    out = model(x)
    print(model.state_dict().keys())
    print(out.shape)
