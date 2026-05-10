#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/workspace/llm_pointseg
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} /root/miniconda3/bin/python train_textproto_opt.py  --save_dir /root/autodl-tmp/workspace/llm_pointseg/outputs/textproto_llm_warm_gate_area5  --epochs 40  --batch_size 8  --num_workers 8  --lr 3e-4  --step_size 20  --text_prototypes /root/autodl-tmp/workspace/llm_pointseg/language_prior/s3dis_clip_text_prototypes.pt  --text_weight 0.0  --learnable_text_gate true  --text_gate_init 0.02  --resume_checkpoint /root/autodl-tmp/workspace/llm_pointseg/outputs/baseline_area5/best.pth  --visual_aux_weight 0.0  --text_aux_weight 0.05 \
  --backbone_enable_flash true
