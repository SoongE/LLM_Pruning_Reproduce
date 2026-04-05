import math
import os
import re
from functools import partial
from pathlib import Path

from safetensors.torch import load_file
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset

HOME = os.getenv("HOME")


def load_state_dict(name, tie_word_embeddings=False):
    MODEL_MAPPING = {
        'meta-llama/Llama-2-7b-hf': f'{HOME}/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9',
        'meta-llama/Llama-2-13b-hf': f'{HOME}/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1',
        'meta-llama/Llama-3.2-3B': f'{HOME}/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062',
        'meta-llama/Llama-3.1-8B': f'{HOME}/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b',
        'lmsys/vicuna-7b-v1.3': f'{HOME}/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.3/snapshots/236eeeab96f0dc2e463f2bebb7bb49809279c6d6',
        'lmsys/vicuna-13b-v1.5': f'',
        'qwen2.5-7B': f'{HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796',
        'qwen2.5-14B': f'{HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B/snapshots/97e1e76335b7017d8f67c08a19d103c0504298c9'
    }

    snapshot_dir = Path(MODEL_MAPPING[name])

    state_dict = {}
    for file_path in sorted(snapshot_dir.glob("model-*.safetensors")):
        state_dict.update(load_file(file_path))

    pop_list = list()
    for k in state_dict.keys():
        if 'model.layers' in k and 'rotary_emb' in k: pop_list.append(k)
    for p in pop_list: state_dict.pop(p)

    if tie_word_embeddings:
        state_dict.update({'lm_head.weight': state_dict['model.embed_tokens.weight']})
    return state_dict


class DictDataset(Dataset):
    def __init__(self, dataset_dict):
        self.dataset_dict = dataset_dict
        self.keys = list(dataset_dict.keys())

    def __len__(self):
        return len(self.dataset_dict[self.keys[0]])

    def __getitem__(self, idx):
        item = dict()

        for key in self.keys:
            item[key] = self.dataset_dict[key][idx]
        return item


def create_criterion(criterion):
    if criterion == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError(f'Not supported criterion: {criterion}')


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        max_learning_rate: float,
        min_learning_rate: float,
        num_cycles: float = 0.5,
        last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_schedule_with_warmup_lr_lambda(
        current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float,
        max_learning_rate: float,
        min_learning_rate: float,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    _lambda = max(0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return (min_learning_rate + _lambda * (max_learning_rate - min_learning_rate)) / max_learning_rate


def get_layer_number(key):
    m = re.search(r"model\.layers\.(\d+)\.", key)
    if m:
        layer = int(m.group(1))
    else:
        layer = None
    return layer
