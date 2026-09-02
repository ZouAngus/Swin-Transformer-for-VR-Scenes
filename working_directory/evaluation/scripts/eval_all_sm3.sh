#!/bin/bash
cd ~/my_mmaction2/work_dirs/long_eval_18cls
PY=~/anaconda3/envs/mmaction2/bin/python
for pkl in pkl_sm3/*.pkl; do
  name=$(basename $pkl .pkl)
  # name pattern: {prefix}_{scene}_{view}
  scene=$(echo $name | cut -d_ -f2)
  if [ -f "eval_sm3/$name/eval_results.json" ]; then continue; fi
  $PY ~/action/tools/eval_long_video.py     --pred-pkl $pkl     --video-name ${name}.mp4     --excel-dir ~/action/data/excel     --scene $scene     --class-ind ~/my_mmaction2/data/myvideo/classInd.txt     --frame-counts ~/action/data/raw_long/frame_counts.json     --window-size 64 --stride 16 --threshold 0.0 --fps 30     --duration-error-threshold 0.1     --out-dir eval_sm3/$name > evalsm_${name}.log 2>&1 || echo "FAIL $name"
done
echo EVAL_ALL_DONE
