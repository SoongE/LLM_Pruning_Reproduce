import multiprocessing as mp
import os
import subprocess
import time

from multi_utils import setup_logger, Style

backbone_dict = {
    '2-7b': 'meta-llama/Llama-2-7b-hf',
    '2-13b': 'meta-llama/Llama-2-13b-hf',
    'v2-7b': 'lmsys/vicuna-7b-v1.3',
    'v2-13b': 'lmsys/vicuna-13b-v1.5',
    '3-3b': 'meta-llama/Llama-3.2-3B',
    '3-8b': 'meta-llama/Llama-3.1-8B',
    "q3-4b": "Qwen/Qwen3-4B-Base",
    "q3-8b": "Qwen/Qwen3-8B-Base",
}

GPUs = '4,5,6,7'
RUNNING_COMMANDS = [
    # Pruning 25%
    ('--num_layers_to_prune 9 --adapter_type lora', 'Qtr_Llama3-3B_LoRA', '3-3b'),
    ('--num_layers_to_prune 9 --adapter_type fourier', 'Qtr_Llama3-3B_FOURIER', '3-3b'),
    ('--num_layers_to_prune 9 --adapter_type adalora', 'Qtr_Llama3-3B_AdaLoRA', '3-3b'),
    ('--num_layers_to_prune 9 --adapter_type vera', 'Qtr_Llama3-3B_VeRA', '3-3b'),
    # # STREAMLINE 50%
    # ('--num_layers_to_prune 17 --adapter_type lora', 'Half_Llama3-3B_LoRA', '3-3b'),
    # ('--num_layers_to_prune 17 --adapter_type fourier', 'Half_Llama3-3B_FOURIER', '3-3b'),
    # ('--num_layers_to_prune 17 --adapter_type adalora', 'Half_Llama3-3B_AdaLoRA', '3-3b'),
    # ('--num_layers_to_prune 17 --adapter_type vera', 'Half_Llama3-3B_VeRA', '3-3b'),
]
commands = [
    (f"python run_peft.py --model_name_or_path {backbone_dict[backbone]} "
     f"{config} --wandb --name {name}")
    for config, name, backbone in RUNNING_COMMANDS
]

eval_datasets = ['winogrande,arc_easy,mathqa,race', 'arc_challenge,openbookqa,piqa,boolq', 'mmlu', 'hellaswag']
eval_commands = [
    (f"python run_lm_eval.py {backbone_dict[backbone]} --adapter "
     f"--name {name} --checkpoint output_peft/{name} {dataset}")
    for dataset in eval_datasets
    for config, name, backbone in RUNNING_COMMANDS
]

logger = setup_logger()
stdout = subprocess.DEVNULL  # Hides standard print() output
stderr = subprocess.DEVNULL  # (Optional) Hides error messages too or tqdm


def run_process(cmd, gpu_id, free_gpu_queue, task_id, total_tasks):
    """
    Worker function to execute a single shell command on a specific GPU.
    """
    try:
        # Log start of the process
        logger.info(f"Starting Task #{task_id}/{total_tasks} on GPU-{gpu_id}", extra={'icon': '🚀', 'color': Style.CYAN})
        logger.info(f"Command: {cmd}", extra={'icon': '  ↳', 'color': Style.BLUE})

        # Clone current environment and set CUDA_VISIBLE_DEVICES
        # This forces the script to see only the assigned GPU ID.
        current_env = os.environ.copy()
        current_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Execute the command (Blocking call)
        # Using shell=True allows full shell syntax (pipes, redirects, etc.)
        subprocess.run(cmd, shell=True, env=current_env, check=True)

        # Log successful completion
        logger.info(f"Task #{task_id} Finished successfully on GPU-{gpu_id}", extra={'icon': '✅', 'color': Style.GREEN})

    except subprocess.CalledProcessError as e:
        # Log failure
        logger.info(f"Task #{task_id} Failed on GPU-{gpu_id} with error: {e}", extra={'icon': '❌', 'color': Style.RED})

    finally:
        free_gpu_queue.put(gpu_id)
        # logger.info(f"GPU-{gpu_id} is now free.", extra={'icon': '♻', 'color': Style.MAGENTA})


def main():
    gpus = GPUs.split(',')
    free_gpu_queue = mp.Queue()
    for i in gpus:
        free_gpu_queue.put(int(i))

    # 2. Process Management
    processes = []
    total_tasks = len(commands)

    logger.info("=" * 60)
    logger.info(f"{Style.BOLD}      GPU JOB SCHEDULER STARTING{Style.RESET}")
    logger.info(f"{Style.BOLD}      Tasks: {total_tasks} | GPUs: {len(gpus)}{Style.RESET}")
    logger.info("=" * 60 + "\n")

    for i, cmd in enumerate(commands):
        task_id = i + 1

        # 3. Wait for an available GPU (Blocking)
        # The script halts here if the queue is empty until a GPU is returned.
        gpu_id = free_gpu_queue.get()

        # 4. Launch the process
        p = mp.Process(target=run_process, args=(cmd, gpu_id, free_gpu_queue, task_id, total_tasks))
        p.start()
        processes.append(p)

        # Slight delay to prevent race conditions in log printing
        time.sleep(0.1)

    # 5. Wait for all child processes to complete
    logger.info("All tasks scheduled. Waiting for completion...", extra={'icon': '💤', 'color': Style.MAGENTA})

    for p in processes:
        p.join()

    logger.info("=" * 60)
    logger.info("ALL TRAIN TASKS COMPLETED!", extra={'icon': '🎉', 'color': Style.GREEN})
    logger.info("=" * 60 + "\n")

    total_tasks = len(eval_commands)
    logger.info("=" * 60)
    logger.info(f"{Style.BOLD}      EVALUATION JOB SCHEDULER STARTING{Style.RESET}")
    logger.info(f"{Style.BOLD}      Tasks: {total_tasks} | GPUs: {len(gpus)}{Style.RESET}")
    logger.info("=" * 60 + "\n")

    for i, cmd in enumerate(eval_commands):
        task_id = i + 1
        gpu_id = free_gpu_queue.get()

        p = mp.Process(target=run_process, args=(cmd, gpu_id, free_gpu_queue, task_id, total_tasks))
        p.start()
        processes.append(p)

        time.sleep(0.1)

    logger.info("All eval tasks scheduled. Waiting for completion...", extra={'icon': '💤', 'color': Style.MAGENTA})
    for p in processes:
        p.join()

    logger.info("\n" + "=" * 60)
    logger.info("ALL EVALUATION TASKS COMPLETED!", extra={'icon': '🎉', 'color': Style.GREEN})
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    main()
