#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/workspace/llm_pointseg
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} /root/miniconda3/bin/python train_textproto_opt.py  --save_dir /root/autodl-tmp/workspace/llm_pointseg/outputs/sota_fullft_flash_lr5e6_from_lr1e5_best_area5_llm_gate  --epochs 80  --batch_size 2  --num_workers 8  --lr 5e-6  --step_size 30  --gamma 0.5  --text_prototypes /root/autodl-tmp/workspace/llm_pointseg/language_prior/s3dis_clip_text_prototypes.pt  --text_weight 0.0  --learnable_text_gate true  --text_gate_init 0.02  --resume_checkpoint /root/autodl-tmp/workspace/llm_pointseg/outputs/sota_fullft_flash_lr1e5_area5_llm_gate/best.pth  --backbone_freeze_encoder false  --backbone_enable_flash true  --text_aux_weight 0.05
