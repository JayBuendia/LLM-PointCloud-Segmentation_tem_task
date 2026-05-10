#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/workspace/llm_pointseg
SRC=/root/autodl-tmp/workspace/llm_pointseg/outputs/sota_fullft_flash_lr5e6_bs8_from_lr1e5_best_area5_llm_gate/best.pth
SAVE=/root/autodl-tmp/workspace/llm_pointseg/outputs/sota_fullft_flash_lr3e6_bs32_from_globalbest_area5_llm_gate
LOG=outputs/logs/sota_fullft_flash_lr3e6_bs32_gpu1_from_globalbest_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$SAVE" outputs/logs
if [ ! -f "$SAVE/best.pth" ]; then
  cp "$SRC" "$SAVE/best.pth"
fi
rm -f "$SAVE/last.pth"
OLD=$(cat outputs/sota_fullft_lr3e6_gpu1.pid 2>/dev/null || true)
if [ -n "$OLD" ]; then kill "$OLD" 2>/dev/null || true; sleep 3; fi
pkill -f "train_utonia_textproto_opt.py.*sota_fullft_flash_lr3e6_area5_llm_gate" 2>/dev/null || true
pkill -f "train_utonia_textproto_opt.py.*sota_fullft_flash_lr3e6_bs32_from_globalbest_area5_llm_gate" 2>/dev/null || true
sleep 2
CUDA_VISIBLE_DEVICES=1 nohup /root/miniconda3/bin/python -u train_utonia_textproto_opt.py  --save_dir "$SAVE"  --epochs 80  --batch_size 32  --num_workers 8  --lr 3e-6  --step_size 30  --gamma 0.5  --text_prototypes /root/autodl-tmp/workspace/llm_pointseg/language_prior/s3dis_clip_text_prototypes.pt  --text_weight 0.0  --learnable_text_gate true  --text_gate_init 0.02  --resume_checkpoint "$SAVE/best.pth"  --utonia_freeze_encoder false  --utonia_enable_flash true  --text_aux_weight 0.05  > "$LOG" 2>&1 &
PID=$!
echo "$PID" > outputs/sota_fullft_lr3e6_gpu1.pid
echo "$LOG" > outputs/sota_fullft_lr3e6_gpu1.logpath
echo "started pid=$PID log=$LOG save=$SAVE"
