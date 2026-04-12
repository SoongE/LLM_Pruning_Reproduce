"""
LLaMA3 Layer Pruning + LoRA LM Loss Recovery
=============================================
Steps:
  1. Load LLaMA3 model
  2. Remove N layers (structured pruning)
  3. Attach Adapters
  4. Fine-tune with Causal LM loss for recovery
"""

import logging
import sys
from pathlib import Path

import datasets
import torch
import torch.nn as nn
import transformers
import wandb
from accelerate import Accelerator
from dataclasses import dataclass, field, asdict
from datasets import load_from_disk
from peft import get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    get_cosine_schedule_with_warmup,
    set_seed, AutoConfig,
)
from transformers.data.data_collator import torch_default_data_collator
from transformers.trainer_pt_utils import get_model_param_count

from peft_configs import get_adapter_config
from zinfo import ENTITY

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="[%(levelname)s|%(name)s] %(asctime)s >> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# ────────────────────────────────────────────────────────────────
# Argument Dataclasses
# ────────────────────────────────────────────────────────────────

@dataclass
class InfoArguments:
    name: str = field(default=None, metadata={"help": "Experiment name"})
    wandb: bool = field(default=False, metadata={"help": "Turn on wandb logging"})
    project: str = field(default="LLM_Pruning_Recovery", metadata={"help": "WandB project name"})
    output_dir: str = field(default="output_peft", metadata={"help": "Where to save outputs"})
    should_log: bool = field(default=False, metadata={"help": "Whether to log verbosely"})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None, metadata={"help": "HF hub ID or local path to the base model"})
    adapter_type: str = field(default=None, metadata={"help": "Adapter type"})
    torch_dtype: str = field(default="bfloat16",
                             metadata={"help": "Dtype for model weights: float32 | float16 | bfloat16"})


@dataclass
class PruningArguments:
    num_layers_to_prune: int = field(default=None, metadata={"help": "Number of layers to remove"})
    prune_strategy: str = field(default="last", metadata={"help": "Layer selection strategy: last | first | middle"})


@dataclass
class TrainingArguments:
    batch_size: int = field(default=2, metadata={"help": "Per-device batch size"})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": "Gradient accumulation steps"})
    num_workers: int = field(default=4, metadata={"help": "DataLoader workers"})

    learning_rate: float = field(default=1e-4, metadata={"help": "Peak learning rate"})
    min_learning_rate: float = field(default=1e-5, metadata={"help": "Min LR at end of cosine schedule"})
    weight_decay: float = field(default=1e-2, metadata={"help": "AdamW weight decay"})
    warmup_ratio: float = field(default=0.05, metadata={"help": "Fraction of steps used for LR warmup"})
    num_epochs: int = field(default=20, metadata={"help": "Number of training epochs"})
    seed: int = field(default=42, metadata={"help": "Random seed"})

    logging_steps: int = field(default=50, metadata={"help": "Log every N optimizer steps"})


def select_layers_to_prune(num_total_layer: int, num_prune: int, strategy: str) -> list[int]:
    assert num_prune < num_total_layer, "num_layers_to_prune must be less than total layers."

    if strategy == "last":
        keep_last = 1
        end_idx = num_total_layer - keep_last
        start_idx = end_idx - num_prune
        return list(range(start_idx, end_idx))

    elif strategy == "first":
        keep_first = max(1, num_total_layer // 8)
        return list(range(keep_first, keep_first + num_prune))

    elif strategy == "middle":
        mid = num_total_layer // 2
        start = mid - num_prune // 2
        return list(range(start, start + num_prune))

    else:
        raise ValueError(f"Unknown prune strategy: '{strategy}'. Choose from: last | first | middle")


def layer_pruning(model, layers_to_remove: list[int]):
    original_layers = model.model.layers

    remove_set = set(layers_to_remove)
    kept = [layer for i, layer in enumerate(original_layers) if i not in remove_set]

    model.model.layers = nn.ModuleList(kept)
    model.config.num_hidden_layers = len(kept)
    return model


# ────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────
def build_dataset(tokenizer_name):
    tokenizer_name = tokenizer_name.lower()
    if 'llama-2' in tokenizer_name:
        path = 'data/tokenized/finewebedu_llama2_train'
    elif 'llama-3' in tokenizer_name:
        path = 'data/tokenized/finewebedu_llama3_train'
    elif 'qwen3' in tokenizer_name:
        path = 'data/tokenized/finewebedu_qwen3_train'
    else:
        raise ValueError

    ds = load_from_disk(path)
    ds = ds.add_column('label', ds['input_ids'])
    return ds


def run_training(
        model,
        train_dl: DataLoader,
        training_args: TrainingArguments,
        info_args: InfoArguments,
        accelerator: Accelerator,
):
    no_decay = ["bias", "layernorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

    total_steps = (len(train_dl) // training_args.gradient_accumulation_steps) * training_args.num_epochs
    warmup_steps = int(total_steps * training_args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, train_dl, scheduler = accelerator.prepare(model, optimizer, train_dl, scheduler)

    global_step = 0

    for epoch in range(training_args.num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(model):
                batch["labels"] = batch["labels"].long()
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            running_loss += loss.item()

            if global_step > 0 and global_step % training_args.logging_steps == 0:
                avg_loss = running_loss / (step + 1)
                lr = scheduler.get_last_lr()[0]
                logger.debug(f"[Epoch {epoch + 1}] step {global_step} | loss: {avg_loss:.4f} | lr: {lr:.2e}")
                if info_args.wandb:
                    wandb.log({"train/loss": avg_loss, "train/lr": lr}, step=global_step)

        avg_epoch_loss = running_loss / len(train_dl)
        logger.debug(f"[Epoch {epoch + 1}] finished | avg loss: {avg_epoch_loss:.4f}")
        if info_args.wandb:
            wandb.log({"train/epoch_loss": avg_epoch_loss, "epoch": epoch + 1})

    return accelerator.unwrap_model(model)


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    info_args, model_args, pruning_args, training_args = HfArgumentParser(
        (InfoArguments, ModelArguments, PruningArguments, TrainingArguments)).parse_args_into_dataclasses()

    # ── Output dir ──────────────────────────────────────────────
    output_dir = Path(info_args.output_dir) / info_args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging ─────────────────────────────────────────────────
    if info_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        logger.setLevel(logging.DEBUG)
        datasets.utils.logging.set_verbosity(logging.DEBUG)
    transformers.utils.logging.set_verbosity(logging.WARNING)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    set_seed(training_args.seed)

    # ── WandB ────────────────────────────────────────────────────
    if info_args.wandb:
        wandb.init(
            project=info_args.project,
            entity=ENTITY,
            name=info_args.name,
            config={
                "info_args": asdict(info_args),
                "model_args": asdict(model_args),
                "pruning_args": asdict(pruning_args),
                "training_args": asdict(training_args),
            },
            settings=wandb.Settings(_disable_stats=True),
        )

    # ── Accelerator ──────────────────────────────────────────────
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    torch_dtype = (
        config.torch_dtype if model_args.torch_dtype == 'auto' else getattr(torch, model_args.torch_dtype))
    config.torch_dtype = torch_dtype
    model_args.torch_dtype = torch_dtype

    mixed_precision = {
        torch.float32: "no",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }.get(model_args.torch_dtype, "no")

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    # ── Dataset ──────────────────────────────────────────────────
    logger.debug("[1/5] Building dataset")
    dataset = build_dataset(model_args.model_name_or_path)
    train_dl = DataLoader(
        dataset,
        batch_size=training_args.batch_size,
        shuffle=True,
        collate_fn=torch_default_data_collator,
        num_workers=training_args.num_workers,
        pin_memory=True,
    )
    logger.debug(f"  Dataset size: {len(dataset):,} chunks")

    # ── Model & Tokenizer ────────────────────────────────────────
    logger.debug(f"[2/5] Loading model: {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=model_args.torch_dtype,
        device_map="auto",
    )
    logger.debug(f"  Original layers: {model.config.num_hidden_layers}")

    # ── Pruning ──────────────────────────────────────────────────
    logger.debug(f"[3/5] Pruning {pruning_args.num_layers_to_prune} layers ({pruning_args.prune_strategy})")
    layers_to_remove = select_layers_to_prune(
        model.config.num_hidden_layers,
        pruning_args.num_layers_to_prune,
        pruning_args.prune_strategy,
    )
    model = layer_pruning(model, layers_to_remove)

    pruned_base_dir = output_dir / "pruned_base"
    model.save_pretrained(pruned_base_dir)
    logger.debug(f"  Pruned base saved → {pruned_base_dir}")

    # ── Attach PEFT Adapter ──────────────────────────────────────
    logger.debug("[4/5] Attaching PEFT adapters")
    adapter_config = get_adapter_config(model_args.adapter_type, total_step=(len(train_dl) // training_args.gradient_accumulation_steps) * training_args.num_epochs)
    model = get_peft_model(model, adapter_config)

    logger.debug(f"  Total params     : {get_model_param_count(model, trainable_only=False) / 1e6:.2f}M")
    logger.debug(f"  Trainable params : {get_model_param_count(model, trainable_only=True) / 1e6:.2f}M")

    # ── Training ─────────────────────────────────────────────────
    logger.debug("[5/5] Starting LoRA LM recovery training")
    model = run_training(model, train_dl, training_args, info_args, accelerator)

    # ── Save LoRA adapter ────────────────────────────────────────
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    logger.debug(f"✅ LoRA adapter saved → {adapter_dir}")

    if info_args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
