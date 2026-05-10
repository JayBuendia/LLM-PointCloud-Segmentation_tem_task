#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/workspace/llm_pointseg
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} /root/miniconda3/bin/python train_textproto.py \
  --save_dir /root/autodl-tmp/workspace/llm_pointseg/outputs/textproto_area5_tw01 \
  --epochs 60 \
  --batch_size 8 \
  --num_workers 8 \
  --text_prototypes /root/autodl-tmp/workspace/llm_pointseg/language_prior/s3dis_clip_text_prototypes.pt \
  --text_weight 0.1 \
  --visual_aux_weight 0.0 \
  --text_aux_weight 0.0 \
  --backbone_enable_flash true
