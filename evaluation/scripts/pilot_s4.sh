#!/bin/bash
cd ~/my_mmaction2
PY=~/anaconda3/envs/mmaction2/bin/python
vids=(11_boss_C 12_travel_C 13_gallery_L 15_bowling_C)
for i in 0 1 2 3; do
  v=${vids[$i]}
  p=$(echo $v | cut -d_ -f1)
  CUDA_VISIBLE_DEVICES=$i $PY tools/infer_long_video.py     --video ~/action/data/raw_long/$p/$v.mp4     --config configs/recognition/swin/my_swin.py     --checkpoint checkpoints/bg_7289/best_acc_top1_epoch_27.pth     --num-classes 18 --stride 4     --out work_dirs/long_eval_18cls/pkl_s4/$v.pkl > work_dirs/long_eval_18cls/pilot_$v.log 2>&1 &
done
wait
for v in ${vids[@]}; do
  scene=$(echo $v | cut -d_ -f2)
  $PY ~/action/tools/eval_long_video.py --pred-pkl work_dirs/long_eval_18cls/pkl_s4/$v.pkl     --video-name $v.mp4 --excel-dir ~/action/data/excel --scene $scene     --class-ind ~/my_mmaction2/data/myvideo/classInd.txt     --frame-counts ~/action/data/raw_long/frame_counts.json     --window-size 64 --stride 4 --threshold 0.0 --fps 30 --duration-error-threshold 0.1     --out-dir work_dirs/long_eval_18cls/eval_s4/$v > /dev/null 2>&1
done
echo PILOT_DONE
