import os
from typing import Any

import torch
from accelerate import init_empty_weights
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.trainer_pt_utils import get_model_param_count

from ..decoder_module import DecoderLM
from ..sharded_dataset import SafeTensorShards
from ..utils import DictDataset, get_layer_number, load_state_dict


def make_streamline_layer(config_name, config, best_layer):
    layer = DecoderLM(config_name, config, best_layer, 1024)
    return layer


def make_streamline_loader(dataset_root, input_layer_idx, output_layer_idx, batch_size, num_workers) -> tuple[
    DataLoader[Any], DictDataset]:
    ids_path = os.path.join(dataset_root, f'manifest_layer{input_layer_idx}.json')
    ods_path = os.path.join(dataset_root, f'manifest_layer{output_layer_idx}.json')

    ds1 = SafeTensorShards(ids_path)
    ds2 = SafeTensorShards(ods_path)

    ds = DictDataset({'input_features': ds1, 'output_features': ds2})
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    return dl, ds


def make_streamline_deploy(config_name, best_layer, layer_interval, checkpoint_dir):
    config = AutoConfig.from_pretrained(config_name)
    config.num_hidden_layers = config.num_hidden_layers - layer_interval

    state_dict = load_state_dict(config_name, tie_word_embeddings=config.tie_word_embeddings)

    merge_idx_start = best_layer
    merge_idx_end = best_layer + layer_interval
    pop_list = list()
    rename_list = list()
    for key in state_dict.keys():
        layer_idx = get_layer_number(key)
        if layer_idx is None: continue

        if layer_idx in range(merge_idx_start + 1, merge_idx_end + 1):
            pop_list.append(key)
        elif layer_idx > merge_idx_end:
            rename_list.append((layer_idx, key))

    for p in pop_list: state_dict.pop(p)
    for layer_idx, key in rename_list:
        new_key = key.replace(f'.{layer_idx}.', f'.{layer_idx - layer_interval}.')
        state_dict[new_key] = state_dict.pop(key)

    layer_state_dict = torch.load(os.path.join(checkpoint_dir, f'layer_{best_layer}.pth'), weights_only=True)
    state_dict.update(layer_state_dict)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    model.load_state_dict(state_dict, assign=True)
    if config.tie_word_embeddings:
        model.tie_weights()

    return model


def check_model_size(config_name, layer_interval):
    # Llama3-8B: 8.03 | 6.07(24.4%, 20/9) | 4.10(48.9%, 18)
    # Llama3-3B: 3.61 | 2.70(25%, 17/9) | 1.79(50.4%, 18)
    # Llama2-7B: 6.74 | 5.12(24%, 21/8) | 3.30(51%, 17)
    config = AutoConfig.from_pretrained(config_name)
    config.num_hidden_layers = config.num_hidden_layers - layer_interval

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    model = model.to_empty(device="cpu")

    print(f"Number of trainable parameters = {get_model_param_count(model, trainable_only=True) / 1e9:.2f}B")


if __name__ == '__main__':
    config_name = 'meta-llama/Llama-2-7b-hf'
    # make_streamline_deploy(config_name, 19, 8, 'output/streamline_llama3_8b_1')
    check_model_size(config_name, layer_interval=17)
