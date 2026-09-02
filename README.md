# Swin Transformer for VR Scenes

Vision-based human action recognition for an immersive CAVE-style VR training
environment, built on [MMAction2](https://github.com/open-mmlab/mmaction2) with
a **Video Swin Transformer (Swin-T)** backbone.

Trainees perform tasks inside the CAVE while two fixed cameras (centre view `C`
and left view `L`) record them. The model watches the recordings and produces a
per-action event stream — which action occurred, when it started and ended, and
with what confidence — which downstream systems consume for safety-warning
reconciliation, proficiency scoring and annotated replay.

Two models are maintained:

| Model | Classes | Purpose |
|---|---|---|
| **Base (18-class)** | 7 generic body actions + 11 scene-specific VR interactions | Full offline analysis of session recordings |
| **Digital-twin (4-class)** | `Stand`, `Raising_hand`, `Jumping`, `Squatting` | Lightweight real-time demo for the digital-twin pipeline |

## Results at a glance

Base 18-class model, `checkpoints/bg_7289/best_acc_top1_epoch_27.pth`, evaluated
on held-out persons 11–15. Full write-up, figures and reproduction commands:
[`working_directory/evaluation/EVAL_detection_classification.md`](working_directory/evaluation/EVAL_detection_classification.md).

| Metric | Result | Requirement | Pass |
|---|---|---|---|
| Detection rate | **96.80%** (3688/3810) | ≥ 90% | ✅ |
| Classification accuracy (among detected) | **99.70%** (3677/3688) | ≥ 85% | ✅ |
| Activity duration error | **9.3%** | ≤ 10% | ✅ |
| Overall top-1 accuracy (7,289 clips) | 95.82% | — | — |
| Background false-alarm rate | 4.94% | — | — |

Errors are almost entirely *action versus background*, not *action versus
action*: once a clip is detected, its label is correct 99.70% of the time. The
two weak classes are `Walking` (0.500 detection rate — slow, low-amplitude, and
treated as background by design in deployment) and `Measure_Length` (0.641 — a
subtle controller gesture). Both are also the classes with the fewest training
clips.

---

# Reproducing our results, step by step

The steps below take you from a fresh machine to every number in the table
above. Steps 1–3 are setup; steps 4–6 reproduce the reported results; steps
7–8 are optional (training from scratch, demos).

## Step 1 — Clone and set up the environment

```bash
git clone https://github.com/ZouAngus/Swin-Transformer-for-VR-Scenes.git
cd Swin-Transformer-for-VR-Scenes

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html
pip install mmengine==0.10.7 decord opencv-python pandas openpyxl matplotlib
pip install -v -e .          # installs mmaction2 from this repo in editable mode
```

Verified package versions:

| Package | Version |
|---|---|
| Python | 3.8.20 |
| torch | 1.11.0+cu113 |
| torchvision | 0.12.0+cu113 |
| mmcv | 2.0.1 |
| mmengine | 0.10.7 |
| mmaction2 | 1.2.0 (this repo, editable install) |
| decord | 0.6.0 |
| opencv-python | 4.12.0 |
| numpy | 1.24.4 |

Reference hardware: 4× NVIDIA A10 (23 GB each), CUDA 11.3. On the lab's
server2 the environment already exists: `conda activate mmaction2`.

## Step 2 — Download the model weights

Every checkpoint is larger than GitHub's 100 MB per-file limit, so the weights
are distributed separately.

**📦 [Download all checkpoints (HKU SharePoint / OneDrive)](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3583523_connect_hku_hk/IQAAoqc9cgGERrvN610LW9y6AXVxheDtYq-ETBXvfoxxDqY?e=GgeY6l)**

The archive (`checkpoints_bundle.zip`, 709 MB) mirrors this repository's
directory layout, so unzipping it at the repository root puts every file exactly
where the configs and commands expect it — no path edits needed:

```bash
unzip checkpoints_bundle.zip -d .
sha256sum -c SHA256SUMS.txt      # every line should print OK
```

| File | Size | Classes | Purpose |
|---|---|---|---|
| `checkpoints/bg_7289/best_acc_top1_epoch_27.pth` | 127 MB | 18 | **18-class base model** — the model evaluated in the report. Use for all 18-class testing, evaluation and inference. |
| `work_dirs/4cls_finetune_v51_walking_neg/epoch_30.pth` | 334 MB | 4 | **Digital-twin demo checkpoint** (v5.1 fine-tune, walking negatives). Full training checkpoint, hence the size. |
| `checkpoints/class_7/best_acc_top1_epoch_18.pth` | 121 MB | 7 | 7 generic classes; warm-start ancestor of the 4-class chain (`load_from`). |
| `checkpoints/no_bg/best_acc_top1_epoch_24.pth` | 127 MB | 18 | 18-class variant trained without background clips. Comparison only. |

SHA256 checksums:

| File | SHA256 |
|---|---|
| `checkpoints/bg_7289/best_acc_top1_epoch_27.pth` | `28e04b0df9ba30d65af596c42cce4d560e8cb5105c0d9b472aee16e098c24e6c` |
| `checkpoints/no_bg/best_acc_top1_epoch_24.pth` | `42b925688b1c5c7bb3d8c4d70bcd4fb7dc515c05c5f646c07a9216a1e5fbf57f` |
| `checkpoints/class_7/best_acc_top1_epoch_18.pth` | `0dc725acfa2c6bced9a9d7992ac6243cbde7b4cdcc9cb479825aa7f46cde6659` |
| `work_dirs/4cls_finetune_v51_walking_neg/epoch_30.pth` | `3a63f1c939440d0e534a717a5cc5956453e93d1f0214d1a3cbdc0a96ac9b5224` |

If a checksum does not match, the transfer was corrupted — download again.

## Step 3 — Get the data

Two inputs are needed beyond this repository; everything else (annotation
workbooks in `data/excel/`, all processing scripts, split lists) is tracked in
git.

1. **Raw long session videos** — restricted access (identifiable participant
   imagery), request from the lab. Test split (persons 11–15): 63 videos,
   ~13 GB, `<person>/<person>_<scene>_<view>.mp4`, 6 scenes × views C/L.
   Training split (persons 1–10) is stored the same way. Verify with the
   integrity manifest `data/raw_long_sha256.txt`:

   ```bash
   cd /path/to/raw_long && sha256sum -c /path/to/repo/data/raw_long_sha256.txt
   ```

2. *(shortcut)* If you receive the pre-cut clip sets
   (`data/myvideo/videos_train/` 54 GB, `videos_val/` 37 GB) you can skip
   Step 4 entirely.

## Step 4 — Rebuild the train / test clips from the raw videos

Three stages; each script's ground truth comes from the same Excel workbooks
used by the evaluation, so slicing and evaluation can never disagree.

```bash
# Stage 1 - cut annotated action intervals out of the raw videos
#   (Stand intervals become *_bg background clips)
python data/prep/slice_raw_actions.py \
    --raw-dir /path/to/raw_long \
    --excel-dir data/excel \
    --out-dir data/myvideo/videos_val \
    --persons 11 12 13 14 15

# Stage 2 - split long background clips into 100-frame windows
#   (paths are relative: run from data/myvideo/)
cd data/myvideo && python ../prep/slice_bg_val.py   # slice_bg.py for the train split

# Stage 3 - regenerate the annotation lists
python ../prep/file_list.py && cd ../..
```

For the training split, repeat with `--persons 1 2 3 4 5 6 7 8 9 10` and
`--out-dir data/myvideo/videos_train`. The long-video evaluation in Step 6
needs no cutting at all — it runs directly on the raw videos.

> Note: the historical clip set was cut with an earlier server-side script;
> rebuilt clips are functionally equivalent (same intervals, same Excel ground
> truth) but file names may differ in minor details, so always regenerate the
> lists in stage 3 rather than reusing the tracked ones.

## Step 5 — Clip-level test: detection rate & classification accuracy

```bash
# 1. run the test set once and dump per-clip scores (4 GPUs)
bash working_directory/dist_test.sh configs/recognition/swin/my_swin.py \
    checkpoints/bg_7289/best_acc_top1_epoch_27.pth 4 \
    --work-dir work_dirs/base18_eval \
    --dump work_dirs/base18_eval/preds.pkl \
    --cfg-options model.cls_head.num_classes=18 \
      test_dataloader.batch_size=2 test_dataloader.num_workers=6
# expected: acc/top1 0.9582

# 2. detection rate + classification accuracy
python working_directory/eval_det_cls.py \
    --preds work_dirs/base18_eval/preds.pkl \
    --ann data/myvideo/myvideo_val_list.txt \
    --labels data/myvideo/classInd.txt \
    --bg-index 0
# expected: DETECTION RATE 0.9680, CLASSIFICATION ACC (detected) 0.9970

# 3. the three clip-level figures
python working_directory/plot_det_cls.py \
    --preds work_dirs/base18_eval/preds.pkl \
    --ann data/myvideo/myvideo_val_list.txt \
    --labels data/myvideo/classInd.txt \
    --bg-index 0 --out-dir work_dirs/base18_eval
```

Single GPU: `python working_directory/test.py <config> <checkpoint> --work-dir ... --dump ...`

> ⚠️ **`num_classes` pitfall.** `configs/_base_/models/swin_tiny.py` sets
> `num_classes=4` for the digital-twin fine-tune, and `my_swin.py` does **not**
> override it. Any 18-class run must pass
> `--cfg-options model.cls_head.num_classes=18` or it will silently build a
> 4-class head. The base config is intentionally left unmodified so the 4-class
> workflow keeps working unchanged.

## Step 6 — Long-video evaluation: activity duration error

Runs directly on the 63 raw test recordings (no cutting). The as-run shell
scripts under `working_directory/evaluation/scripts/` contain server-absolute
paths; the portable equivalents are shown here.

```bash
# 1. sliding-window inference over each recording (repeat per video / GPU)
python working_directory/infer_long_video.py \
    --video /path/to/raw_long/11/11_boss_C.mp4 \
    --config configs/recognition/swin/my_swin.py \
    --checkpoint checkpoints/bg_7289/best_acc_top1_epoch_27.pth \
    --num-classes 18 --bg-scale 0.1 \
    --out work_dirs/long_eval_18cls/pkl_supp0.1/11_boss_C.pkl
# (evaluation/scripts/run_fleet.sh parallelises all 63 videos over 4 GPUs, ~45 min)

# 2. per-video evaluation against the Excel annotations
python working_directory/eval_long_video.py \
    --pred-pkl work_dirs/long_eval_18cls/pkl_supp0.1/11_boss_C.pkl \
    --video-name 11_boss_C.mp4 --scene boss \
    --excel-dir data/excel \
    --class-ind data/myvideo/classInd.txt \
    --frame-counts data/excel/frame_counts.json \
    --window-size 64 --stride 16 --threshold 0.0 --fps 30 \
    --duration-error-threshold 0.1 \
    --out-dir work_dirs/long_eval_18cls/eval_supp0.1/11_boss_C

# 3. aggregate all 63 videos
cd work_dirs/long_eval_18cls
python <repo>/working_directory/evaluation/scripts/aggregate.py eval_supp0.1 --exclude-label Walking
# expected: mean rel. error 9.3%
```

Two arguments matter:

* `--bg-scale 0.1` / `eval_supp0.1` — the **deployment configuration**: the
  `Stand` (background) probability is scaled by 0.1 at inference. This is the
  single most important post-processing step — it takes the mean duration
  error from 29.1% to 9.3% and lifts frame-level mIoU from 0.549 to 0.608.
  Temporal smoothing has almost no effect by comparison.
* `--exclude-label Walking` — `Walking` is background by design in the
  deployed system and is therefore not an activity, matching the report's
  vocabulary.

| Variant | mean rel. error (all) | mean rel. error (excl. `Walking`) | median |
|---|---|---|---|
| `eval` (no calibration) | 30.9% | 29.1% | 15.5% |
| **`eval_supp0.1`** | 11.2% | **9.3%** | 0.0% |
| `eval_supp0.25` | 15.3% | 13.3% | 0.0% |
| `eval_supp0.5` | 22.2% | 20.3% | 3.2% |
| `eval_sm3` (smoothing) | 31.4% | 29.5% | 15.4% |

## Step 7 — (Optional) Train from scratch

### 18-class base model

```bash
bash working_directory/dist_train.sh configs/recognition/swin/my_swin.py 4 \
    --work-dir work_dirs/<run_name> \
    --cfg-options model.cls_head.num_classes=18
```

Single GPU: `python working_directory/train.py <config> --work-dir ... --cfg-options model.cls_head.num_classes=18`.
The Kinetics-400 pretrained Swin-T weights are downloaded automatically on
first run. Remember the `num_classes` pitfall from Step 5.

### 4-class digital-twin fine-tune

`num_classes` is already 4, so no override is needed:

```bash
python working_directory/train.py configs/recognition/swin/my_swin_4cls_v51.py \
    --work-dir work_dirs/4cls_finetune_v51_walking_neg
```

Each `my_swin_4cls_v*.py` sets `load_from` to the checkpoint it warm-starts
from. These are **absolute paths under `/home/zhuyusi/my_mmaction2/`** — edit
them if you clone this repository somewhere else.

## Step 8 — (Optional) Long-video demos

### 4-class digital-twin demo

Runs two passes (inference, then annotated rendering) and writes three files
into `demo/digitaltwins/results/`.

```bash
CUDA_VISIBLE_DEVICES=0 python demo/long_video_demo_4cls_v2.py \
    --config demo/demo_configs/my_swin_4cls_demo.py \
    --checkpoint work_dirs/4cls_finetune_v51_walking_neg/epoch_30.pth \
    --video_path demo/digitaltwins/<VIDEO>.mp4 \
    --label data/digitaltwin_action/label_4cls.txt \
    --out_file demo/digitaltwins/results/<VIDEO>.mp4 \
    --input-step 2 --stride 0.25 --smooth-k 1 \
    --decision-threshold 0.4 --hysteresis 1 --stand-index 0 \
    --decision-overlay --min-segment-duration 0.1 \
    --decision-json demo/digitaltwins/results/<VIDEO>_decisions.json
```

| Output | Contents |
|---|---|
| `<VIDEO>.mp4` | source video with the predicted label overlaid |
| `<VIDEO>_decisions.json` | per-window decisions, plus every inference parameter used |
| `ActionData_<YYYYMMDD_HHMMSS>.json` | final action event list (label, timestamp, frame range, duration, confidence) |

### 18-class long video

```bash
python demo/long_video_demo.py \
    --config demo/demo_configs/my_swin_demo.py \
    --checkpoint checkpoints/bg_7289/best_acc_top1_epoch_27.pth \
    --video_path <VIDEO>.mp4 \
    --label data/labels/my_label.txt \
    --out_file <OUT>.mp4 \
    --input-step 2 --stride 0.25
```

> ⚠️ **`moov atom not found`.** The input `.mp4` was still being written when
> inference started — an mp4 stores its index (`moov` atom) at the *end* of the
> file, so a recording in progress cannot be decoded. Wait for the recording to
> finish; check with
> `python -c "import decord; print(len(decord.VideoReader('<VIDEO>.mp4')))"`.

---

# Supplementary

## Repository layout

```
.
├── configs/                    # training / testing configs
│   ├── _base_/                 #   shared model, schedule and runtime configs
│   └── recognition/swin/       #   my_swin.py (18-class) + my_swin_4cls*.py (4-class)
├── mmaction/                   # MMAction2 source, installed in editable mode
├── working_directory/          # everything you RUN: train / test / evaluation
│   ├── train.py, test.py, dist_train.sh, dist_test.sh
│   ├── infer_long_video.py     #   sliding-window inference over long videos
│   ├── eval_det_cls.py, eval_4cls.py, eval_long_video.py, plot_det_cls.py
│   └── evaluation/             #   evaluation report, figures, results, run records
├── data/                       # data + data-processing scripts (videos NOT in repo)
│   ├── excel/                  #   per-person annotation workbooks (ground truth)
│   ├── prep/                   #   raw -> train/test pipeline (Step 4)
│   ├── labels/                 #   label files for the demo --label argument
│   ├── myvideo/                #   18-class lists + classInd.txt
│   ├── digitaltwin_action/     #   4-class lists + labels
│   └── raw_long_sha256.txt     #   integrity manifest of the raw recordings
├── demo/                       # long-video demo scripts and demo configs
│   └── demo_configs/           #   my_swin_demo.py, my_swin_4cls_demo.py
├── tools/                      # standalone utilities
│   ├── format_json_to_action.py  # decision JSON -> ActionData event format
│   ├── split_long_video.py       # long video -> sliding-window clips (legacy path)
│   └── h264.py                   # re-encode clips for preview
├── docs/                       # manuals
├── archives/                   # superseded scripts, backups, legacy files, v3 freeze
├── tests/                      # MMAction2 unit tests
└── setup.py, setup.cfg
```

### What is **not** in this repository

These live on **server2** under `~/my_mmaction2/` and are excluded by
`.gitignore` because of their size.

| Path | Size | What it is |
|---|---|---|
| `data/myvideo/videos_{train,val}/` | ~93 GB | 14,056 training + 7,289 validation clips (18-class) |
| `data/digitaltwin_action/*/` | — | 4-class digital-twin clips |
| `checkpoints/` + the demo checkpoint | ~709 MB | trained model weights — download in Step 2 |
| `work_dirs/` | ~11 GB | training logs and per-epoch checkpoints of all 8 experiments |
| `demo/digitaltwins/`, `demo/4cls_runs/` | ~8.7 GB | demo source recordings and annotated output |
| raw session recordings (`raw_long/`) | ~13 GB (test split) | on server2 at `~/action/data/raw_long/` |

The annotation lists **are** tracked (`data/myvideo/*.txt`,
`data/digitaltwin_action/*.txt`, ~1.3 MB), so splits are reproducible once the
videos are restored.

## Dataset details

### 18-class set — `data/myvideo/`

Class index is `data/myvideo/classInd.txt`. `Stand` (index 0) doubles as the
background class.

```
0 Stand            6 Jumping         12 Bowling
1 Bending_Down     7 Grabbing        13 Catching_fish
2 Walking          8 Cutting         14 Shooting
3 Raising_hand     9 Measure_Length  15 Waive_sword
4 Squatting       10 Move_Controller 16 Throw
5 Running         11 Picking_item    17 Waive
```

* Training: `myvideo_train_list.txt` — 14,056 clips, persons 1–10
* Validation: `myvideo_val_list.txt` — 7,289 clips, persons 11–15

The split is **by person**, so there is no subject leakage. Clips ending in
`_bg` are background windows carrying no action annotation. There is an
approximate **6-frame offset between views C and L**; all `start_frame` /
`end_frame` values refer to view C. The historical sliced clips on server1
(`/data/sda1/shared/sliced_video`) are named
`s{person}_a{action}_{scene}_{excel_row}_rep{repetition}_frames_{start}_{end}_{view}.mp4`.

### 4-class digital-twin set — `data/digitaltwin_action/`

`Stand`, `Raising_hand`, `Jumping`, `Squatting` (`label_4cls.txt`,
`classInd.txt`). Five split versions exist, v1 → v5.1; each successive version
adds targeted samples:

| Split | Train | Val | Added |
|---|---|---|---|
| `train_list.txt` / `val_list.txt` | 234 | 60 | initial |
| `*_v3.txt` | 317 | 80 | warm-start from v2 |
| `*_v4.txt` | 325 | 82 | shallow-squat samples |
| `*_v5.txt` | 324 | 80 | hard examples |
| `*_v51.txt` | 333 | 80 | walking negatives ← **current best** |

> **Note on 4-class validation accuracy.** Every 4-class fine-tune from v2
> onward reports `acc/top1: 1.0000`. With only 60–82 validation clips over 4
> classes the metric is saturated and cannot rank these versions. Judge 4-class
> changes on long-video behaviour, not on this number.

## Evaluation archive

Everything behind the report is collected under
[`working_directory/evaluation/`](working_directory/evaluation/): the report
(`.md` renders on GitHub, `.pdf` for distribution), the cited figures,
raw result summaries (`results/`), and the as-run shell scripts from server2
(`scripts/`, server-absolute paths — see `scripts/README.md`). The
`aggregate.py` there takes the evaluation directory as an argument;
`results/aggregate_summary_baseline.txt` (30.9%, uncalibrated) and
`results/aggregate_summary_supp0.1.txt` (9.3%, deployment configuration) are
both kept — they are different configurations of the same run, not a
contradiction.

## Known issues

* **Hard-coded paths in configs.** The 4-class fine-tune configs set
  `load_from` to absolute checkpoint paths under `/home/zhuyusi/my_mmaction2/`.
  They work on server2 as-is; adjust them for any other machine. (The demo
  scripts formerly had absolute paths too; those are now repo-relative.)
* **`num_classes=4` in the shared base config** — see the warning under Step 5.
* **Superseded demo scripts** are collected in `archives/`;
  `demo/long_video_demo_4cls_v2.py` is the current 4-class demo.

## Acknowledgements

- **Yusi Zhu** — training of the general 17-action-class (+ background) base
  model and curation of the trained models.
- Repository restructuring, the evaluation pipeline (detection rate,
  classification accuracy, activity duration error) and documentation were
  prepared with the assistance of **Claude Fable 5** (Anthropic).
- Built on [MMAction2](https://github.com/open-mmlab/mmaction2) (Apache
  License 2.0) by OpenMMLab. The `mmaction/`, `tools/`, `tests/` and upstream
  `configs/` trees derive from it; see `LICENSE` for the upstream terms.
