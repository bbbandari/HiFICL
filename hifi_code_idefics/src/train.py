import os

import paths
from data_module import DataModule
from shift_model import ShiftModel, Strategy
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    TQDMProgressBar,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import Callback
import json
import time as _time
from termcolor import colored
import hydra
from omegaconf import DictConfig
from utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(config_path="config", config_name="train.yaml", version_base=None)
def main(cfg: DictConfig):
    def get_max_epochs():
        num_query_samples = cfg.data.num_query_samples
        model_name = cfg.model_name
        if "idefics-9b" in model_name:
            if num_query_samples < 100:
                return 15
            if num_query_samples <= 500:
                return 10
            return 10
        elif "idefics2-8b" in model_name:
            if num_query_samples < 100:
                return 15
            if num_query_samples <= 500:
                return 10
            return 5
        elif "llava" in model_name:
            if num_query_samples <= 500:
                return 10
            return 5

    def save_when(epoch):
        num_query_samples = cfg.data.num_query_samples
        model_name = cfg.model_name
        if "idefics-9b" in model_name:
            if num_query_samples < 100:
                return epoch >= 10
            if num_query_samples <= 200:
                if cfg.data.name == "coco":
                    return epoch >= 5
                return epoch >= 7
            if num_query_samples <= 500:
                return epoch >= 5
            return epoch >= 5
        elif "idefics2-8b":
            if num_query_samples <= 100:
                return epoch >= 9
            if num_query_samples <= 200:
                return epoch >= 4
            if num_query_samples <= 300:
                return epoch >= 2
            if num_query_samples <= 500:
                return epoch >= 1
            return True
        elif "llava" in model_name:
            if num_query_samples <= 1000:
                return epoch >= 5
            return True

    max_epochs = cfg.epochs if cfg.epochs else get_max_epochs()
    runname = get_expand_runname(cfg)
    print(colored(f"Training for {runname} on {cfg.model_name}", "light_blue"))

    if cfg.resume:
        save_dir = os.path.join(paths.result_dir, "ckpt", runname)
        os.makedirs(save_dir, exist_ok=True)
        exist_ckpt_epochs = [
            int(d.split("-")[-1])
            for d in os.listdir(save_dir)
            if os.path.exists(os.path.join(save_dir, d)) and d.split("-")[-1].isdigit()
        ]
        for i in range(max_epochs):
            if save_when(i) and i not in exist_ckpt_epochs:
                break
        else:
            print(f"All checkpoints {runname} matched, skip...")
            return
    pl.seed_everything(cfg.data.seed)
    os.makedirs(paths.result_dir, exist_ok=True)
    wb_logger = WandbLogger(
        save_dir=paths.result_dir,
        name=runname,
        project="VQAInContextVector",
        log_model=False,
    )
    torch.set_float32_matmul_precision("medium")
    class TrainStatsCallback(Callback):
        def __init__(self, run_name: str, cfg: DictConfig, world_size: int):
            self.run_name = run_name
            self.cfg = cfg
            self.world_size = world_size
            self.step_start_time = None
            self.stats_path = os.path.join(
                paths.result_dir,
                "record",
                get_expand_runname(cfg),
                "train_stats.jsonl",
            )

        def on_train_start(self, trainer, pl_module):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            # Ensure directory exists on rank 0
            if getattr(trainer, "global_rank", 0) == 0:
                os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self.step_start_time = _time.perf_counter()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if self.step_start_time is None:
                return
            step_time_s = max(_time.perf_counter() - self.step_start_time, 1e-9)
            # Estimate throughput as samples per second across all devices
            # Use configured per-device batch size times world size
            per_device_batch_size = int(self.cfg.batch_size)
            global_batch_size = per_device_batch_size * max(self.world_size, 1)
            samples_per_sec = global_batch_size / step_time_s

            gpu_mem = None
            gpu_mem_peak = None
            if torch.cuda.is_available():
                try:
                    device = torch.cuda.current_device()
                    gpu_mem = torch.cuda.memory_allocated(device)
                    gpu_mem_peak = torch.cuda.max_memory_allocated(device)
                except Exception:
                    pass

            record = {
                "epoch": int(trainer.current_epoch),
                "batch_idx": int(batch_idx),
                "step_time_s": float(step_time_s),
                "throughput_samples_per_s": float(samples_per_sec),
                "gpu_memory_bytes": int(gpu_mem) if gpu_mem is not None else None,
                "gpu_peak_memory_bytes": int(gpu_mem_peak) if gpu_mem_peak is not None else None,
            }

            if getattr(trainer, "global_rank", 0) == 0:
                os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
                with open(self.stats_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) if "CUDA_VISIBLE_DEVICES" in os.environ else 1

    trainer = pl.Trainer(
        logger=wb_logger,
        callbacks=[
            LearningRateMonitor(),
            TQDMProgressBar(refresh_rate=10),
            TrainStatsCallback(runname, cfg, world_size),
        ],
        # fast_dev_run=True,
        # devices=1,
        max_epochs=max_epochs,
        devices=len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")),
        use_distributed_sampler=False,
        strategy=cfg.strategy,
        precision=cfg.precision,
        gradient_clip_val=cfg.grad_clip_val,
        log_every_n_steps=2,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        enable_checkpointing=False,
    )

    lmm = build_model(cfg)
    convert_to_peft(cfg, lmm)
    data_module = DataModule(cfg, lmm)
    shift_encoder = hydra.utils.instantiate(cfg.encoder.cls, _partial_=True)(lmm=lmm)

    model = ShiftModel(
        cfg,
        shift_encoder,
        eval(cfg.encoder.model_strategy),
        save_checkpoint_when=save_when,
    )
    trainer.fit(
        model,
        data_module,
    )


if __name__ == "__main__":
    main()
