import logging
import os
import sys
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import datasets
import torch
import torchmetrics
import transformers
import wandb
from accelerate import Accelerator
from dataclasses import dataclass, field, asdict
from transformers import (
    AutoConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_pt_utils import get_model_param_count

from streamline import make_streamline_layer, make_streamline_loader
from utils import get_cosine_schedule_with_warmup, create_criterion
from zinfo import ENTITY


def configure_tf32(enable: bool = True) -> None:
    has_new_matmul_api = hasattr(torch.backends.cuda.matmul, "fp32_precision")
    has_new_cudnn_api = hasattr(torch.backends.cudnn, "conv") and hasattr(
        torch.backends.cudnn.conv, "fp32_precision"
    )

    if has_new_matmul_api:
        torch.backends.cuda.matmul.fp32_precision = "tf32" if enable else "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = enable
        torch.set_float32_matmul_precision("high" if enable else "highest")

    if has_new_cudnn_api:
        torch.backends.cudnn.conv.fp32_precision = "tf32" if enable else "ieee"
    else:
        torch.backends.cudnn.allow_tf32 = enable


configure_tf32()

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="[%(levelname)s|%(name)s] %(asctime)s >> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


@dataclass
class InfoArguments:
    name: str = field(default=None, metadata={"help": "Experiment Name"})
    wandb: bool = field(default=False, metadata={"help": "Turn on wandb."})
    project: str = field(default='LLM_Pruning_Reproduce', metadata={"help": "The project name of wandb"})
    output_dir: str = field(default='output', metadata={"help": "Where to save output"})
    should_log: bool = field(default=False, metadata={"help": "Whether to log"})


@dataclass
class ModelArguments:
    config_name: str = field(default=None, metadata={"help": "Which config to use"})


@dataclass
class TrainingArguments:
    criterion: str = field(default='mse', metadata={"help": "Select loss function"})
    epochs: int = field(default=10, metadata={"help": "Number of epochs"})
    torch_dtype: str = field(default="auto", metadata={"help": "Which dtype to use"})

    dataset: str = field(default=None, metadata={"help": "Which dataset to use"})
    batch_size: int = field(default=8, metadata={"help": "Batch size"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of accumulation steps"})
    num_workers: int = field(default=4, metadata={"help": "Number of workers"})

    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate"})
    min_learning_rate: float = field(default=1e-5, metadata={"help": "Min learning rate"})
    weight_decay: float = field(default=1e-2, metadata={"help": "Weight decay"})
    seed: int = field(default=42, metadata={"help": "Random seed"})
    target_layer: int = field(default=None, metadata={"help": "Which layer of model to use"})


@dataclass
class StreamlineArguments:
    streamline: bool = field(default=False, metadata={"help": "Whether to streamline"})
    best_layer: int = field(default=None, metadata={"help": "Which layer of model to use"})
    layer_intervals: int = field(default=None, metadata={"help": "Number of intervals to merge"})


def main():
    info_args, model_args, training_args, streamline_args = HfArgumentParser(
        (InfoArguments, ModelArguments, TrainingArguments, StreamlineArguments)).parse_args_into_dataclasses()
    output_dir = Path(os.path.join(info_args.output_dir, info_args.name))
    if not output_dir.exists(): output_dir.mkdir(parents=True, exist_ok=True)

    # Setup Logging
    if info_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(logging.WARNING)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    set_seed(training_args.seed)

    if info_args.wandb:
        config = {
            'info_args': asdict(info_args),
            'model_args': asdict(model_args),
            'training_args': asdict(training_args),
        }
        wandb.init(project=info_args.project, entity=ENTITY, config=config, name=f"{info_args.name}",
                   settings=wandb.Settings(_disable_stats=True))

    # Setup Model
    config = AutoConfig.from_pretrained(model_args.config_name)
    torch_dtype = (
        config.torch_dtype if training_args.torch_dtype == 'auto' else getattr(torch, training_args.torch_dtype))
    config.torch_dtype = torch_dtype
    training_args.torch_dtype = torch_dtype

    if streamline_args.streamline:
        model = make_streamline_layer(model_args.config_name, config, streamline_args.best_layer)
        training_args.target_layer = streamline_args.best_layer
    else:
        raise ValueError(f'Supported reproduce model is not selected.')

    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    trainable_parameter_names = [n for n, p in model.named_parameters() if p.requires_grad]
    logger.debug(f"Total Model Size: {n_params / 1e6:.2f}M")
    logger.debug(f"Number of trainable parameters = {get_model_param_count(model, trainable_only=True) / 1e6:.2f}M")
    logger.debug(f"Trainable Parameters: {trainable_parameter_names}")

    # Setup Datasets
    if streamline_args.streamline:
        train_dl, train_ds = make_streamline_loader(training_args.dataset + '_train',
                                                    streamline_args.best_layer - 1,
                                                    streamline_args.best_layer + streamline_args.layer_intervals,
                                                    training_args.batch_size,
                                                    training_args.num_workers)

        val_dl, _ = make_streamline_loader(training_args.dataset + '_test',
                                           streamline_args.best_layer - 1,
                                           streamline_args.best_layer + streamline_args.layer_intervals,
                                           training_args.batch_size,
                                           training_args.num_workers)
    else:
        raise ValueError('Supported reproduce model is not selected.')

    # Setup Criterion, Optimizer, Scheduler
    criterion = create_criterion(training_args.criterion)

    no_decay = ["bias", "layernorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate,
                                  weight_decay=training_args.weight_decay)

    total_steps = len(train_dl) * training_args.epochs // training_args.gradient_accumulation_steps
    if training_args.min_learning_rate > training_args.learning_rate:
        training_args.min_learning_rate = training_args.learning_rate * 0.1

    # Soong: Official Streamline use half warmup but, in our experiments, conventional cosine scheduling works well.
    # if streamline_args.streamline:
    #     scheduler = get_cosine_schedule_with_warmup(
    #         optimizer=optimizer,
    #         num_warmup_steps=total_steps * 0.01 * 0.5,
    #         num_training_steps=total_steps * 0.5,
    #         max_learning_rate=training_args.learning_rate,
    #         min_learning_rate=training_args.min_learning_rate,
    #     )
    # else:
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=total_steps * 0.01,
        num_training_steps=total_steps,
        max_learning_rate=training_args.learning_rate,
        min_learning_rate=training_args.min_learning_rate,
    )

    if info_args.wandb:
        wandb.config.update({
            'training_args': asdict(training_args),
            'dataset_size': len(train_ds)
        }, allow_val_change=True)

    # Setup Accelerator
    if config.torch_dtype == torch.float32 or config.torch_dtype == torch.float:
        mixed_precision = 'no'
    elif config.torch_dtype == torch.float16:
        mixed_precision = 'fp16'
    elif config.torch_dtype == torch.bfloat16:
        mixed_precision = 'bf16'
    else:
        raise ValueError()
    accelerator = Accelerator(gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                              mixed_precision=mixed_precision)
    model, optimizer, train_dl, val_dl, scheduler = accelerator.prepare(model, optimizer, train_dl, val_dl, scheduler)

    losses = torchmetrics.MeanMetric().to(accelerator.device)
    logging_iter = len(train_dl) // 16

    for epoch in range(training_args.epochs):
        # Train
        model.train()
        losses.reset()
        s = perf_counter()
        for i, item in enumerate(train_dl):
            x = item['input_features']
            y = item['output_features']
            with accelerator.accumulate(model):
                out_features = model(x)
                loss = criterion(out_features, y.to(torch.float32))
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            losses.update(loss)

            duration = perf_counter() - s
            s = perf_counter()
            if i % logging_iter == 0:
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                nb_remain = len(train_dl) - i - 1
                nb_remain += (training_args.epochs - epoch - 1) * len(train_dl)
                eta_seconds = duration * nb_remain

                loss_value = losses.compute().item()
                logger.debug(f'{"Train":>5}: {epoch:>3} [{i:>4d}/{len(train_dl) - 1}] '
                             f'({100. * i / (len(train_dl) - 1):>3.0f}%)]  '
                             f'Loss: {loss_value:#.3g}  '
                             f'LR: {lr:.3e}  '
                             f'TP: {training_args.batch_size / duration:>7.2f}/s  '
                             f'ETA: {timedelta(seconds=int(eta_seconds))}  '
                             )
                if info_args.wandb: wandb.log({'loss': loss_value, 'lr': lr})

        # Validation
        model.eval()
        losses.reset()
        for i, item in enumerate(val_dl):
            x = item['input_features']
            y = item['output_features']
            with torch.no_grad():
                out_features = model(x)
                loss = criterion(out_features, y.to(torch.float32))
            losses.update(loss)
        if info_args.wandb: wandb.log({'val_loss': losses.compute().item()})

    state_dict = accelerator.unwrap_model(model).state_dict()
    torch.save(state_dict, os.path.join(output_dir, f'layer_{training_args.target_layer}.pth'))

    if info_args.wandb: wandb.finish()


if __name__ == "__main__":
    main()
