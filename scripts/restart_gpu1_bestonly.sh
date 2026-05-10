#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/workspace/llm_pointseg
SAVE=/root/autodl-tmp/workspace/llm_pointseg/outputs/sota_fullft_flash_lr3e6_area5_llm_gate
LOG=outputs/logs/sota_fullft_flash_lr3e6_gpu1_bestonly_restart_$(date +%Y%m%d_%H%M%S).log
mkdir -p outputs/logs
rm -f "$SAVE/last.pth"
OLD=$(pgrep -f "train_textproto_opt.py.*sota_fullft_flash_lr3e6_area5_llm_gate" | head -n 1 || true)
if [ -n "$OLD" ]; then
  echo "stopping old gpu1 python $OLD"
  kill "$OLD" 2>/dev/null || true
  sleep 5
fi
CUDA_VISIBLE_DEVICES=1 nohup /root/miniconda3/bin/python -u train_textproto_opt.py  --save_dir "$SAVE"  --epochs 120  --batch_size 2  --num_workers 8  --lr 3e-6  --step_size 40  --gamma 0.5  --text_prototypes /root/autodl-tmp/workspace/llm_pointseg/language_prior/s3dis_clip_text_prototypes.pt  --text_weight 0.0  --learnable_text_gate true  --text_gate_init 0.02  --resume_checkpoint "$SAVE/best.pth"  --backbone_freeze_encoder false  --backbone_enable_flash true  --text_aux_weight 0.05  > "$LOG" 2>&1 &
PID=$!
echo "$PID" > outputs/sota_fullft_lr3e6_gpu1.pid
echo "$LOG" > outputs/sota_fullft_lr3e6_gpu1.logpath
echo "started pid=$PID log=$LOG"
sleep 15
ps -fp "$PID" || true
tail -n 80 "$LOG" || true
