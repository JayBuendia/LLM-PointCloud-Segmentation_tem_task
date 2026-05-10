#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/workspace/llm_pointseg
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} /root/miniconda3/bin/python train_textproto.py \
  --save_dir /root/autodl-tmp/workspace/llm_pointseg/outputs/baseline_area5 \
  --epochs 60 \
  --batch_size 8 \
  --num_workers 8 \
  --text_weight 0.0
