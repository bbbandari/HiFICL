import argparse
import os
import re
import subprocess
import sys
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import shlex
import paths


def merge_args(base_args, new_args):
    base_dict = {arg.partition("=")[0]: arg for arg in base_args}

    if new_args:
        new_dict = {arg.partition("=")[0]: arg for arg in new_args}
        base_dict.update(new_dict)

    return list(base_dict.values())


def get_avail_devices(devices, requires_memory=None):
    if not hasattr(get_avail_devices, "cached_requires_memory"):
        if requires_memory is None:
            raise ValueError("requires_memory must be provided on the first call.")
        get_avail_devices.cached_requires_memory = requires_memory

    effective_requires_memory = (
        requires_memory
        if requires_memory is not None
        else get_avail_devices.cached_requires_memory
    )

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    # Handle case when no GPU devices are found
    if result.returncode != 0 or "No devices were found" in result.stdout:
        free_gpus = []
    else:
        free_memory = result.stdout.strip().split("\n")
        free_gpus = [
            str(idx)
            for idx, mem in enumerate(free_memory)
            if mem.strip() and mem.strip().isdigit() and int(mem) > effective_requires_memory
        ]

    if devices:
        return ",".join(set(devices.split(",")) & set(free_gpus))
    else:
        return ",".join(free_gpus)


def run_train(
    runname,
    dataset,
    num_query_sample,
    num_shot,
    model_name,
    train_args,
    devices,
    requires_memory=None,
):
    try:
        process = subprocess.Popen(
            [
                "python",
                "train.py",
            ]
            + merge_args(
                [
                    f"runname={runname}",
                    f"model_name={model_name}",
                    f"data.num_query_samples={num_query_sample}",
                    f"data.name={dataset}",
                    f"data.num_shot={num_shot}",
                ],
                shlex.split(train_args),
            ),
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": get_avail_devices(devices, requires_memory),
            },
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
        )

        returncode = process.wait()

        if returncode != 0:
            if process.stderr and "out of memory" in process.stderr:
                return dataset, num_query_sample, num_shot
        else:
            return True
    except Exception as e:
        print(f"An error occurred during training: {e}", file=sys.stderr)

    return False


def run_eval(
    ckpt_path, dataset, num_query_sample, num_shot, model_name, gpu_id, eval_args, eval_num_shot=0
):
    try:
        process = subprocess.Popen(
            ["python", "eval.py"]
            + merge_args(
                [
                    f"ckpt_path={ckpt_path or 'null'}",
                    f"model_name={model_name}",
                    f"data.name={dataset}",
                    f"data.num_shot={eval_num_shot}",
                    f"data.num_query_samples={num_query_sample}",
                ],
                shlex.split(eval_args),
            ),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id},
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
        )

        returncode = process.wait()

        if returncode != 0:
            if process.stderr and "out of memory" in process.stderr:
                return dataset, num_query_sample, num_shot
            print(
                f"Evaluation failed on GPU {gpu_id} for runname: {ckpt_path}, dataset: {dataset}"
            )
        else:
            return True
    except Exception as e:
        print(f"An error occurred during evaluation: {e}", file=sys.stderr)
    return False


def run_analyze(runname, dataset, num_query_sample, num_shot, model_name, analyze_args):
    if "icl" in runname:
        # runname-model-dataset
        expand_runname = f"{runname}-{model_name}-{dataset}"
    else:
        # runname-model-dataset-training_samples-num_shot
        expand_runname = f"{runname}-{model_name}-{dataset}-{num_query_sample}-{num_shot}shot"
    try:
        subprocess.run(
            ["python", "analyze.py"]
            + merge_args(
                [
                    f"model_name={model_name}",
                    f"record_dir={os.path.join(paths.result_dir, 'record', expand_runname)}",
                    f"data.name={dataset}",
                    f"data.num_shot={num_shot}",
                    f"data.num_query_samples={num_query_sample}",
                ],
                shlex.split(analyze_args),
            ),
            env=os.environ,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(
            f"Analyze failed for runname: {runname}, dataset: {dataset}",
            file=sys.stderr,
        )
        sys.exit(1)


def execute_eval_deepspeed(runname, tasks, model_name, eval_args, devices, eval_num_shot=0):
    """Execute evaluation using DeepSpeed for multi-GPU inference"""
    import subprocess
    import os
    
    # Set environment variables for DeepSpeed
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    
    for task in tasks:
        dataset, num_query_sample, num_shot = task
        
        if "icl" in runname:
            expand_runname = f"{runname}-{model_name}-{dataset}"
            print(f"Running DeepSpeed evaluation: {dataset=}, {num_query_sample=}, {num_shot=}, ICL")
            
            # Run with DeepSpeed
            cmd = [
                "deepspeed", "--num_gpus", str(len(devices.split(","))),
                "eval.py",
                f"ckpt_path=None",
                f"model_name={model_name}",
                f"data.name={dataset}",
                f"data.num_shot={eval_num_shot}",
                f"data.num_query_samples={num_query_sample}",
                f"encoder=mimic_v19",
                f"peft=mimic_v4",
                f"use_deepspeed=True"
            ] + eval_args
            
            print(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error in DeepSpeed evaluation: {result.stderr}")
        else:
            # Handle checkpoint-based evaluation
            ckpt_dir = os.path.join(paths.result_dir, "ckpt", f"{runname}-{model_name}-{dataset}")
            for epoch_ckpt in os.listdir(ckpt_dir):
                if not epoch_ckpt.startswith("epoch-"):
                    continue
                epoch = re.findall(r"\d+", epoch_ckpt)[0]
                ckpt_path = os.path.join(ckpt_dir, epoch_ckpt)
                
                print(f"Running DeepSpeed evaluation: {dataset=}, {num_query_sample=}, {num_shot=}, {epoch=}")
                
                cmd = [
                    "deepspeed", "--num_gpus", str(len(devices.split(","))),
                    "eval.py",
                    f"ckpt_path={ckpt_path}",
                    f"model_name={model_name}",
                    f"data.name={dataset}",
                    f"data.num_shot={eval_num_shot}",
                    f"data.num_query_samples={num_query_sample}",
                    f"encoder=mimic_v19",
                    f"peft=mimic_v4",
                    f"use_deepspeed=True"
                ] + eval_args
                
                print(f"Executing: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error in DeepSpeed evaluation: {result.stderr}")


def execute_eval(runname, tasks, model_name, eval_args, devices, requires_memory=None, use_deepspeed=False, eval_num_shot=0):
    if use_deepspeed and len(devices.split(",")) > 1:
        # Use DeepSpeed for multi-GPU inference
        return execute_eval_deepspeed(runname, tasks, model_name, eval_args, devices, eval_num_shot)
    
    task_queue = tasks.copy()
    futures = {}

    def get_gpu_id():
        nonlocal futures
        available_gpus = [
            gpu_id
            for gpu_id in get_avail_devices(devices, requires_memory).split(",")
            if gpu_id not in futures
        ]
        if not available_gpus:
            # no available GPUs, wait for tasks to finish
            next(as_completed(futures.values())).result()
            available_gpus = [gpu_id for gpu_id, f in futures.items() if f.done()]
            futures = {gpu_id: f for gpu_id, f in futures.items() if not f.done()}

        return available_gpus.pop(0)

    with ThreadPoolExecutor() as executor:

        while task_queue or futures:
            if not task_queue:
                # Wait for running tasks to complete
                next(as_completed(futures.values())).result()
                futures = {gpu_id: f for gpu_id, f in futures.items() if not f.done()}
                continue
            
            dataset, num_query_sample, num_shot = task_queue.pop(0)
            if "icl" in runname:
                # runname-model-dataset
                expand_runname = f"{runname}-{model_name}-{dataset}"
                gpu_id = get_gpu_id()
                print(
                    f"Assigning task to GPU {gpu_id}: {dataset=}, {num_query_sample=}, {num_shot=}, ICL"
                )
                futures[gpu_id] = executor.submit(
                    run_eval,
                    None,
                    dataset,
                    num_query_sample,
                    num_shot,
                    model_name,
                    gpu_id,
                    eval_args,
                    eval_num_shot,
                )
            else:
                # runname-model-dataset-training_samples-num_shot
                expand_runname = f"{runname}-{model_name}-{dataset}-{num_query_sample}-{num_shot}shot"
                ckpt_dir = os.path.join(paths.result_dir, "ckpt", expand_runname)
                for epoch_ckpt in os.listdir(ckpt_dir):
                    # Skip non-epoch directories like config.json
                    if not epoch_ckpt.startswith("epoch-"):
                        continue
                    epoch = re.findall(r"\d+", epoch_ckpt)[0]
                    gpu_id = get_gpu_id()
                    print(
                        f"Assigning task to GPU {gpu_id}: {dataset=}, {num_query_sample=}, {num_shot=}, {epoch=}"
                    )
                    futures[gpu_id] = executor.submit(
                        run_eval,
                        os.path.join(ckpt_dir, epoch_ckpt),
                        dataset,
                        num_query_sample,
                        num_shot,
                        model_name,
                        gpu_id,
                        eval_args,
                        eval_num_shot,
                    )


def main():
    parser = argparse.ArgumentParser(description="Run training and evaluation tasks.")
    parser.add_argument(
        "-r", "--runname", required=True, help="Name for the current run."
    )
    parser.add_argument(
        "-d", "--datasets", required=True, help="Comma-separated list of datasets."
    )
    parser.add_argument(
        "-m", "--model-name", required=True, help="Name of the model to use."
    )
    parser.add_argument(
        "-q",
        "--num-query-samples",
        default="",
        help="Comma-separated list of query samples.",
    )
    parser.add_argument(
        "-s", "--num-shots", required=True, help="Comma-separated list of shots."
    )
    parser.add_argument("-t", "--train", action="store_true", help="Enable train mode.")
    parser.add_argument(
        "-e",
        "--eval",
        action="store_true",
        help="Enable eval mode. If set to true, analyze mode will be also enabled.",
    )
    parser.add_argument(
        "-a", "--analyze", action="store_true", help="Enable analyze mode."
    )
    parser.add_argument(
        "--train-args",
        required=False,
        help="Additional training arguments.",
    )
    parser.add_argument(
        "--eval-args",
        required=False,
        help="Additional evaluation arguments.",
    )
    parser.add_argument(
        "--analyze-args",
        required=False,
        help="Additional evaluation arguments.",
    )
    parser.add_argument(
        "--devices",
        required=False,
        help="Comma-separated list of integers that denotes devices used to train or evaluate.",
    )
    parser.add_argument(
        "--wait-devices-timeout",
        type=int,
        default=0,
        help="Maximum time in minutes to wait for free GPUs. If <= 0, exit immediately if no adequate GPUs are available.",
    )
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="Skip GPU availability check. Use for CPU-only or testing environments.",
    )
    parser.add_argument(
        "--requires_memory",
        type=int,
        default=20000,
        help="The minimal cuda memory used to run train or eval, unit MB.",
    )
    parser.add_argument(
        "--wait-n-devices",
        type=int,
        default=1,
        help="Minimum devices are required to start.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training phase.",
    )
    parser.add_argument(
        "--use-deepspeed",
        action="store_true",
        help="Use multi-GPU inference with torch.distributed (not DeepSpeed).",
    )
    parser.add_argument(
        "--eval-num-shots",
        type=int,
        default=0,
        help="Number of shots to use during evaluation. Default is 0 (no examples, only questions).",
    )
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    num_query_samples = args.num_query_samples.split(",")
    num_shots = args.num_shots.split(",")
    runname = args.runname
    model_name = args.model_name
    train_args = getattr(args, "train_args") or ""
    eval_args = getattr(args, "eval_args") or ""
    analyze_args = getattr(args, "analyze_args") or ""
    devices = args.devices
    timeout = args.wait_devices_timeout
    min_devices = max(args.wait_n_devices, 1)
    requires_memory = args.requires_memory

    if not (args.train | args.eval | args.analyze):
        args.train = args.eval = args.analyze = True
    
    # Handle skip-train option
    if args.skip_train:
        args.train = False

    if not args.skip_gpu_check and (
        not get_avail_devices(devices, requires_memory)  # no even one devices
        or len(get_avail_devices(devices, requires_memory).split(","))
        < min_devices  # less than minimal required devices
    ) and (args.eval or args.train):
        print(f"Cannot find at least {min_devices} devcie(s). Start waiting...")
        for i in range(0, timeout):
            if (
                get_avail_devices(devices, requires_memory)
                and len(get_avail_devices(devices, requires_memory).split(","))
                >= min_devices
            ):
                break
            time.sleep(60)
            print(f"Waited for {min_devices} device(s) for {(i+1)} mins")
        else:
            print(f"Cannot find at least {min_devices} devcie(s). Timeout, exit...")
            return

    if num_query_samples:
        tasks = list(product(datasets, num_query_samples, num_shots))
    else:
        if args.train:
            raise RuntimeError(
                "The option -q/--num_query_samples is required if train mode is enabled."
            )
        tasks = list(product(datasets, num_shots))

    if args.train:
        print("Starting training phase...")
        task_queue = tasks.copy()
        while task_queue:
            dataset, num_query_sample, num_shot = task_queue.pop(0)
            ret = run_train(
                runname,
                dataset,
                num_query_sample,
                num_shot,
                model_name,
                train_args,
                devices,
                requires_memory,
            )

            if isinstance(ret, tuple):
                task_queue.append(ret)
            elif not ret:
                return

    if args.eval:
        print("Starting evaluation phase...")
        execute_eval(runname, tasks, model_name, eval_args, devices, requires_memory, args.use_deepspeed, args.eval_num_shots)

    if args.analyze:
        print("Starting analysis phase...")
        for dataset, num_query_sample, num_shot in tasks:
            run_analyze(
                runname, dataset, num_query_sample, num_shot, model_name, analyze_args
            )


if __name__ == "__main__":
    main()
