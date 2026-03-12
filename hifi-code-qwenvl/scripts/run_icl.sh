#!/bin/bash

runname=$(basename "$0" .sh | sed 's/^run_//')


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/${runname}.log"
mkdir -p "$SCRIPT_DIR"
echo "==== $(date '+%F %T') | START $runname ====" | tee -a "$LOG_FILE"
cd "$SCRIPT_DIR/../src/"

# 修复libstdc++版本冲突问题
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/libstdc++.so.6:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

python pipeline.py \
    -r "icl" \
    -d vqav2 \
    -m Qwen3-VL-8B-Instruct \
    -e \
    -s 8 \
    -q 1000 \
    --requires_memory 20000 \
    --eval-args "ckpt_path=null batch_size=1 iterations=1000 resume=False" \
    --eval-num-shots 8 \
    2>&1 | tee -a "$LOG_FILE"
echo "==== $(date '+%F %T') | END   $runname ====" | tee -a "$LOG_FILE"