#!/bin/bash

runname=$(basename "$0" .sh | sed 's/^run_//')


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/${runname}.log"
mkdir -p "$SCRIPT_DIR"
echo "==== $(date '+%F %T') | START $runname ====" | tee -a "$LOG_FILE"
cd "$SCRIPT_DIR/../src/"

WANDB_MODE=disabled python -u pipeline.py \
    -r "$runname-idev2-r-16" \
    -d vqav2,ok_vqa,coco \
    -m Qwen3-VL-8B-Instruct \
    -q 8000 \
    -s 0 \
    --devices 0,1,2,3 \
    --requires_memory 40000 \
    --wait-devices-timeout 100000 \
    --train-args "encoder=lora peft=lora training.batch_size=2 training.accumulate_grad_batches=8" \
    --eval-args "encoder=lora peft=lora eval.batch_size=8"