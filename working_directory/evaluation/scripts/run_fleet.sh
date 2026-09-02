#!/bin/bash
GPU=$1
cd ~/my_mmaction2
PY=~/anaconda3/envs/mmaction2/bin/python
i=0
for v in $(ls ~/action/data/raw_long/1[1-5]/*.mp4 | sort); do
  if [ $((i % 4)) -eq $GPU ]; then
    name=$(basename $v .mp4)
    out=work_dirs/long_eval_18cls/pkl/${name}.pkl
    if [ ! -f "$out" ]; then
      echo "[GPU$GPU] $name start $(date +%H:%M:%S)"
      CUDA_VISIBLE_DEVICES=$GPU $PY tools/infer_long_video.py --video "$v" --config configs/recognition/swin/my_swin.py --checkpoint checkpoints/bg_7289/best_acc_top1_epoch_27.pth --num-classes 18 --out "$out" >> work_dirs/long_eval_18cls/gpu${GPU}.log 2>&1
      echo "[GPU$GPU] $name done $(date +%H:%M:%S)"
    fi
  fi
  i=$((i+1))
done
echo "[GPU$GPU] ALL DONE"
