import os
from datetime import timedelta
from typing import Annotated

from streamline.utils import make_streamline_deploy

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import torch
import typer
from accelerate import InitProcessGroupKwargs, Accelerator
from accelerate.utils import get_max_memory
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typer import Argument, Option

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.utils import make_table, load_yaml_config

app = typer.Typer(pretty_exceptions_enable=False)

torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)

HELP_PANEL_NAME_1 = "Common Parameters"
HELP_PANEL_NAME_2 = "Model Parameters"
HELP_PANEL_NAME_3 = "Tokenizer Parameters"
HELP_PANEL_NAME_4 = "Evaluation Parameters"


def save_to_csv(name, param, results, file_name='results.csv'):
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write(f"Param(B),Model,Task,Version,N-shot,Metric,Value,Stderr\n")

    if "groups" in results:
        column_name = 'groups'
    else:
        column_name = 'results'

    keys = results[column_name].keys()
    with open(file_name, "+a") as file:
        for key in keys:
            dic = results[column_name][key]
            version = results["versions"].get(key, "N/A")
            n_shot = str(results.get("n-shot", " ").get(key, 0))

            if "alias" in dic:
                key = dic.pop("alias")

            metric_items = dic.items()
            metric_items = sorted(metric_items)

            for (mf), v in metric_items:
                m, _, f = mf.partition(",")
                if m.endswith("_stderr"):
                    continue

                if m + "_stderr" + "," + f in dic:
                    se = dic[m + "_stderr" + "," + f]
                    file.write(f"{param},{name},{key},{version},{n_shot},{m},{v},{se}\n")


def task_manage(tasks):
    task_manager = TaskManager()
    task_list = tasks.split(",")
    task_names = task_manager.match_tasks(task_list)
    for task in [task for task in task_list if task not in task_names]:
        if os.path.isfile(task):
            config = load_yaml_config(task)
            task_names.append(config)
    task_missing = [
        task for task in task_list if task not in task_names and "*" not in task
    ]  # we don't want errors if a wildcard ("*") task name was used

    if task_missing:
        missing = ", ".join(task_missing)
        raise ValueError(
            f"Tasks not found: {missing}. Try `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
        )
    return task_names


def set_parallelization_kwargs(accelerator, max_memory_per_gpu=0.95):
    num_local_processes = accelerator.num_processes
    kwargs = dict()
    max_memory_all_gpus = get_max_memory()
    if "cpu" in max_memory_all_gpus:
        del max_memory_all_gpus["cpu"]

    max_memory_per_gpu_map = {
        k: v * max_memory_per_gpu
        for k, v in max_memory_all_gpus.items()
        if k % num_local_processes
           == (accelerator.process_index % num_local_processes)
    }
    kwargs["max_memory"] = max_memory_per_gpu_map
    kwargs["device_map"] = "auto"
    kwargs["offload_folder"] = "./offload"

    return kwargs


@app.command()
def main(
        model_name: Annotated[str, Argument(help="Name of transformers model or your custom model)")],
        tasks: Annotated[str, Argument(help="Comma-separated list of tasks to evaluate on.")],
        # Model Parameters
        attn: Annotated[
            str, Option(
                help="Attention mechanism for transformers models. Select ['eager', 'sdpa', 'flash_attention_2'].",
                rich_help_panel=HELP_PANEL_NAME_2)] = "flash_attention_2",
        dtype: Annotated[
            str, Option(help="Torch dtype of model.", rich_help_panel=HELP_PANEL_NAME_2)] = "auto",
        parallel: Annotated[
            bool, Option(help="Implement model parallelization when the model is too large to import on single GPU.",
                         rich_help_panel=HELP_PANEL_NAME_2)] = False,
        trust_remote_code: Annotated[
            bool, Option(help="Compile model for fast inference.", rich_help_panel=HELP_PANEL_NAME_2)] = True,
        checkpoint: Annotated[str, Option(help="Checkpoint of safetensors", rich_help_panel=HELP_PANEL_NAME_2)] = None,
        mixed_precision: Annotated[str, Option(help="Mixed precision type. [no | fp8 | fp16 | bf16]",
                                               rich_help_panel=HELP_PANEL_NAME_2)] = None,
        # Tokenizer Parameters
        tokenizer_name: Annotated[
            str, Option(help="Name of tokenizer. If none, it will be model_name.",
                        rich_help_panel=HELP_PANEL_NAME_3)] = None,
        max_length: Annotated[
            int, Option(help="Maximum number of generated tokens.",
                        rich_help_panel=HELP_PANEL_NAME_3)] = None,
        fast: Annotated[
            bool, Option(help="Use fast tokenizer.", rich_help_panel=HELP_PANEL_NAME_3)] = True,
        # Evaluation Parameters
        num_fewshot: Annotated[
            int, Option(help="Number of few shot samples", rich_help_panel=HELP_PANEL_NAME_4)] = 0,
        batch_size: Annotated[
            int, Option(help="Number of batch size", rich_help_panel=HELP_PANEL_NAME_4)] = 1,
        log_samples: Annotated[
            bool, Option(
                help="Write out all model outputs and documents for per-sample measurement and post-hoc analysis",
                rich_help_panel=HELP_PANEL_NAME_4)] = False,
        # Common Parameters
        name: Annotated[
            str, Option(help="Experimental name to save the results.", rich_help_panel=HELP_PANEL_NAME_1)] = None,
        save: Annotated[
            bool, Option(help="Whether to save the experimental results", rich_help_panel=HELP_PANEL_NAME_1)] = True,
        # Streamline Parameters
        streamline: Annotated[bool, Option(help='Whether to streamline')] = False,
        best_layer: Annotated[int, Option(help='Which layer of model to use')] = None,
        layer_interval: Annotated[int, Option(help='Number of intervals to merge')] = None,

):
    if mixed_precision is None:
        config = AutoConfig.from_pretrained(model_name if tokenizer_name is None else tokenizer_name)
        if config.torch_dtype == torch.float32 or config.torch_dtype == torch.float:
            mixed_precision = 'no'
        elif config.torch_dtype == torch.float16:
            mixed_precision = 'fp16'
        elif config.torch_dtype == torch.bfloat16:
            mixed_precision = 'bf16'
        else:
            raise ValueError()

    accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
    accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs], mixed_precision=mixed_precision)

    if parallel:
        parallel_kwargs = set_parallelization_kwargs(accelerator)
    elif attn == "flash_attention_2":
        parallel_kwargs = {"device_map": "cpu"}
    else:
        parallel_kwargs = {}

    if streamline:
        model = make_streamline_deploy(model_name, best_layer, layer_interval, checkpoint)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn,
            torch_dtype=dtype,
            **parallel_kwargs,
        )

    model = accelerator.prepare_model(model)
    model.eval()

    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    print(f"Model Size: {n_params / 1e6:.2f}M")

    tokenizer_name = tokenizer_name or model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=fast, model_max_length=max_length, legacy=False)

    if tasks.endswith('.txt'):
        with open(tasks, 'r') as f:
            tasks = f.readlines()[0]

    gen_kwargs = model.generation_config.to_diff_dict()
    if 'pad_token_id' in gen_kwargs:
        gen_kwargs.pop('pad_token_id')

    results = evaluator.simple_evaluate(
        model=HFLM(pretrained=model, tokenizer=tokenizer, accelerator=accelerator, backend="causal"),
        tasks=task_manage(tasks),
        verbosity="WARNING",
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        log_samples=log_samples,
        gen_kwargs=gen_kwargs,
        # confirm_run_unsafe_code=trust_remote_code
    )

    if accelerator.is_main_process:
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

        if save:
            name = name or model_name
            save_to_csv(name, f"{n_params / 1e9:.2f}", results)

    accelerator.wait_for_everyone()
    return


if __name__ == '__main__':
    app()
