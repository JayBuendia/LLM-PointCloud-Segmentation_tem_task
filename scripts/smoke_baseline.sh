#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/workspace/llm_pointseg
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} /root/miniconda3/bin/python train_utonia_textproto.py \
  --save_dir /root/autodl-tmp/workspace/llm_pointseg/outputs/smoke_baseline \
  --epochs 1 \
  --batch_size 1 \
  --num_workers 0 \
  --limit_train_batches 2 \
  --limit_val_batches 2 \
  --text_weight 0.0 \
  --utonia_enable_flash true
