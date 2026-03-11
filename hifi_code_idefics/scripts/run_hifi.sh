#!/bin/bash

runname=$(basename "$0" .sh | sed 's/^run_//')


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/${runname}.log"
mkdir -p "$SCRIPT_DIR"
echo "==== $(date '+%F %T') | START $runname ====" | tee -a "$LOG_FILE"
cd "$SCRIPT_DIR/../src/"


WANDB_MODE=disabled python -u pipeline.py \
    -r "$runname" \
    -d vqav2 \
    -m idefics2-8b-base \
    -q 1000 \
    -s 8 \
    --devices 0 \
    --requires_memory 35000 \
    --wait-devices-timeout 100000 \
    --skip-gpu-check \
    --eval-num-shots 0 \
    --train-args "encoder=hifi peft=hifi epochs=3" \
    --eval-args "encoder=hifi peft=hifi batch_size=1" \
    2>&1 | tee -a "$LOG_FILE"
echo "==== $(date '+%F %T') | END   $runname ====" | tee -a "$LOG_FILE"