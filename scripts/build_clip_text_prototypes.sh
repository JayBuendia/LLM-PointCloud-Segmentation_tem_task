#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/workspace/llm_pointseg
/root/miniconda3/bin/python language_prior/build_text_prototypes.py \
  --descriptions language_prior/s3dis_descriptions.json \
  --output language_prior/s3dis_clip_text_prototypes.pt \
  --encoder clip \
  --model_name ViT-B/32
