#!/bin/bash
cd ~/my_mmaction2/work_dirs/long_eval_18cls
PY=~/anaconda3/envs/mmaction2/bin/python
for A in 0.5 0.25 0.1; do
  $PY - <<PYEOF
import pickle, glob, os, numpy as np
a=float("$A")
os.makedirs(f"pkl_supp{a}", exist_ok=True)
for fp in glob.glob("pkl/*.pkl"):
    d=pickle.load(open(fp,"rb"))
    out=[]
    for x in d:
        s=x["pred_score"].copy(); s[0]*=a
        out.append({"pred_score":s})
    pickle.dump(out, open(f"pkl_supp{a}/"+os.path.basename(fp),"wb"))
PYEOF
  for pkl in pkl_supp$A/*.pkl; do
    name=$(basename $pkl .pkl); scene=$(echo $name | cut -d_ -f2)
    $PY ~/action/tools/eval_long_video.py --pred-pkl $pkl --video-name ${name}.mp4       --excel-dir ~/action/data/excel --scene $scene       --class-ind ~/my_mmaction2/data/myvideo/classInd.txt       --frame-counts ~/action/data/raw_long/frame_counts.json       --window-size 64 --stride 16 --threshold 0.0 --fps 30 --duration-error-threshold 0.1       --out-dir eval_supp$A/$name > /dev/null 2>&1
  done
  echo "ALPHA $A done"
done
echo SWEEP_DONE
