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

---

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
│   ├── prep/                   #   raw -> train/test pipeline (see "From raw data")
│   ├── labels/                 #   label files for the demo --label argument
│   ├── myvideo/                #   18-class lists + classInd.txt
│   └── digitaltwin_action/     #   4-class lists + labels
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
| `checkpoints/` + the demo checkpoint | ~709 MB | trained model weights — **[available for download](#model-weights)** |
| `work_dirs/` | ~11 GB | training logs and per-epoch checkpoints of all 8 experiments |
| `demo/digitaltwins/`, `demo/4cls_runs/` | ~8.7 GB | demo source recordings and annotated output |
| `~/action/data/{excel,raw_long}/` | — | long-video annotations (Excel) and raw session recordings |

The annotation lists **are** tracked (`data/myvideo/*.txt`,
`data/digitaltwin_action/*.txt`, ~1.3 MB), so splits are reproducible once the
videos are restored.

---

## Model weights

Every checkpoint is larger than GitHub's 100 MB per-file limit, so the weights
are distributed separately.

**📦 [Download all checkpoints (HKU SharePoint / OneDrive)](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3583523_connect_hku_hk/IgAQ9OH1E8knSaKnLVYCFGexASMQNBKG_0cQ79n-CZz4JvU?e=WbB5y1)**

The archive (`checkpoints_bundle.zip`, 709 MB) mirrors this repository's
directory layout, so unzipping it at the repository root puts every file exactly
where the configs and commands expect it — no path edits needed:

```bash
unzip checkpoints_bundle.zip -d /path/to/Swin-Transformer-for-VR-Scenes/
```

### What is in the archive

| File | Size | Classes | Purpose |
|---|---|---|---|
| `checkpoints/bg_7289/best_acc_top1_epoch_27.pth` | 127 MB | 18 | **18-class base model**, trained with background frames included. This is the model evaluated in the report — detection rate 96.80%, classification accuracy 99.70%, duration error 9.3%. Use this for all 18-class training, testing and long-video inference. |
| `work_dirs/4cls_finetune_v51_walking_neg/epoch_30.pth` | 334 MB | 4 | **Used for the digitaltwins demo.** The v5.1 fine-tune (walking negatives), and the checkpoint the `demo/long_video_demo_4cls_v2.py` command below runs with. Larger than the others because it is a full training checkpoint — it carries optimizer state alongside the weights. |
| `checkpoints/class_7/best_acc_top1_epoch_18.pth` | 121 MB | 7 | 7 generic body classes. Warm-start weight loaded via `load_from` in `configs/recognition/swin/my_swin_4cls.py`; the whole 4-class fine-tune chain descends from it. |
| `checkpoints/no_bg/best_acc_top1_epoch_24.pth` | 127 MB | 18 | 18-class variant trained only on action-annotated frames, with no background clips. Kept for comparison; not used in the reported results. |

The archive also contains `CHECKPOINTS_README.md` and `SHA256SUMS.txt`.

### Verifying the download

```bash
unzip checkpoints_bundle.zip -d ./bundle && cd ./bundle
sha256sum -c SHA256SUMS.txt      # every line should print OK
```

| File | SHA256 |
|---|---|
| `checkpoints/bg_7289/best_acc_top1_epoch_27.pth` | `28e04b0df9ba30d65af596c42cce4d560e8cb5105c0d9b472aee16e098c24e6c` |
| `checkpoints/no_bg/best_acc_top1_epoch_24.pth` | `42b925688b1c5c7bb3d8c4d70bcd4fb7dc515c05c5f646c07a9216a1e5fbf57f` |
| `checkpoints/class_7/best_acc_top1_epoch_18.pth` | `0dc725acfa2c6bced9a9d7992ac6243cbde7b4cdcc9cb479825aa7f46cde6659` |
| `work_dirs/4cls_finetune_v51_walking_neg/epoch_30.pth` | `3a63f1c939440d0e534a717a5cc5956453e93d1f0214d1a3cbdc0a96ac9b5224` |

If a checksum does not match, the transfer was corrupted — download again rather
than trying to use the file.

### Which checkpoint goes with which command

| Task | Checkpoint |
|---|---|
| 18-class training / testing / `working_directory/infer_long_video.py` | `checkpoints/bg_7289/best_acc_top1_epoch_27.pth` |
| **digitaltwins long-video demo** (`demo/long_video_demo_4cls_v2.py`) | `work_dirs/4cls_finetune_v51_walking_neg/epoch_30.pth` |
| 4-class fine-tune warm start | `checkpoints/class_7/best_acc_top1_epoch_18.pth` (set automatically by `load_from`) |

---

## Environment

Use the existing `mmaction2` conda environment on server2:

```bash
conda activate mmaction2
# or, without activating:  PATH=~/anaconda3/envs/mmaction2/bin:$PATH
```

To rebuild it elsewhere, these are the versions this code is verified against:

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

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html
pip install mmengine==0.10.7 decord opencv-python
pip install -v -e .          # installs mmaction2 from this repo in editable mode
```

Reference hardware: 4× NVIDIA A10 (23 GB each), CUDA 11.3.

---

## Dataset

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

* Training: `data/myvideo/myvideo_train_list.txt` — 14,056 clips, persons 1–10
* Validation: `data/myvideo/myvideo_val_list.txt` — 7,289 clips, persons 11–15

The split is **by person**, so there is no subject leakage. Clips ending in
`_bg` are background windows carrying no action annotation.

Raw sliced videos (the source the clips were cut from) live on **server1** at
`/data/sda1/shared/sliced_video`, named:

```
s{person}_a{action}_{scene}_{excel_row}_rep{repetition}_frames_{start}_{end}_{view}.mp4
```

There is an approximate **6-frame offset between views C and L**; all
`start_frame` / `end_frame` values refer to view C.

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

---

## From raw data to train / test data

Everything below the raw recordings is reproducible from this repository. A
third party needs exactly two inputs:

1. **Raw long session videos** (~13 GB for the test split, persons 11-15:
   63 videos, `<person>/<person>_<scene>_<view>.mp4`, 6 scenes x views C/L) —
   restricted access, request from the lab. The training split (persons 1-10)
   is stored the same way.
   Integrity manifest: `data/raw_long_sha256.txt` (63 files).

2. **This repository** — the annotation workbooks (`data/excel/`) and all
   processing scripts are tracked in git.

Then run the three-stage pipeline:

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
cd data/myvideo && python ../../data/prep/file_list.py
```

For the training split, repeat with `--persons 1 2 3 4 5 6 7 8 9 10` and
`--out-dir data/myvideo/videos_train`. The long-video (event-level) evaluation
needs no cutting at all — it runs directly on the raw videos (see
`evaluation/`). Note: the historical clip set was cut with an earlier
server-side script; rebuilt clips are functionally equivalent (same intervals,
same Excel ground truth) but file names may differ in minor details, so always
regenerate the lists in stage 3 rather than reusing the tracked ones.

## Training

### 18-class base model

```bash
cd ~/my_mmaction2
python working_directory/train.py configs/recognition/swin/my_swin.py \
    --work-dir work_dirs/<run_name> \
    --cfg-options model.cls_head.num_classes=18
```

Multi-GPU:

```bash
bash working_directory/dist_train.sh configs/recognition/swin/my_swin.py 4 \
    --work-dir work_dirs/<run_name> \
    --cfg-options model.cls_head.num_classes=18
```

> ⚠️ **`num_classes` pitfall.** `configs/_base_/models/swin_tiny.py` sets
> `num_classes=4` for the digital-twin fine-tune, and `my_swin.py` does **not**
> override it. Any 18-class run must pass
> `--cfg-options model.cls_head.num_classes=18` or it will silently build a
> 4-class head. The base config is intentionally left unmodified so the 4-class
> workflow keeps working unchanged.

### 4-class digital-twin fine-tune

`num_classes` is already 4, so no override is needed:

```bash
python working_directory/train.py configs/recognition/swin/my_swin_4cls_v51.py \
    --work-dir work_dirs/4cls_finetune_v51_walking_neg
```

Each `my_swin_4cls_v*.py` sets `load_from` to the checkpoint it warm-starts
from. These are **absolute paths under `/home/zhuyusi/my_mmaction2/`** — edit
them if you clone this repository somewhere else.

---

## Testing

Checkpoint: `checkpoints/bg_7289/best_acc_top1_epoch_27.pth` — see
[Model weights](#model-weights) for the download.

```bash
cd ~/my_mmaction2
bash working_directory/dist_test.sh configs/recognition/swin/my_swin.py \
    checkpoints/bg_7289/best_acc_top1_epoch_27.pth 4 \
    --work-dir work_dirs/base18_eval \
    --dump work_dirs/base18_eval/preds.pkl \
    --cfg-options model.cls_head.num_classes=18 \
      test_dataloader.batch_size=2 test_dataloader.num_workers=6
```

Single GPU: `python working_directory/test.py <config> <checkpoint> --work-dir ... --dump ...`

---

## Long-video inference (demo)

### 4-class digital-twin demo

Checkpoint: `work_dirs/4cls_finetune_v51_walking_neg/epoch_30.pth` — see
[Model weights](#model-weights) for the download.

This is the command used for the digital-twin recordings. It runs two passes
(inference, then annotated rendering) and writes three files into
`demo/digitaltwins/results/`.

```bash
cd ~/my_mmaction2
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

Outputs:

| File | Contents |
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

Batch sliding-window inference over a whole session recording, dumping scores
to a `.pkl` for evaluation:

```bash
python working_directory/infer_long_video.py \
    --video <VIDEO>.mp4 \
    --config configs/recognition/swin/my_swin.py \
    --checkpoint checkpoints/bg_7289/best_acc_top1_epoch_27.pth \
    --num-classes 18 \
    --out work_dirs/long_eval_18cls/pkl/<name>.pkl
```

> ⚠️ **`moov atom not found`.** This means the input `.mp4` was still being
> written when inference started — an mp4 stores its index (`moov` atom) at the
> *end* of the file, so a recording in progress cannot be decoded. The run will
> complete with 0 frames and produce an empty result. Wait for the recording to
> finish and check first:
>
> ```bash
> python -c "import decord; print(len(decord.VideoReader('<VIDEO>.mp4')))"
> ```
>
> If that prints a frame count, the file is complete and safe to process.

---

## Evaluation

Everything needed to reproduce the numbers in the report is collected under
[`evaluation/`](evaluation/):

```
evaluation/
├── EVAL_detection_classification.md   # full report (figures render on GitHub)
├── EVAL_detection_classification.pdf
├── figures/                           # the figures the report cites
├── results/
│   ├── eval_det_cls.txt               # clip-level detection / classification
│   ├── aggregate_summary_baseline.txt # long-video, no background calibration
│   └── aggregate_summary_supp0.1.txt  # long-video, deployment configuration → 9.3%
└── scripts/
    ├── base18_test.sh                 # clip-level test run
    ├── run_fleet.sh                   # per-video sliding-window inference across 4 GPUs
    ├── eval_all.sh                    # per-video evaluation
    ├── eval_all_sm3.sh                # variant: temporal smoothing k=3
    ├── sweep_supp.sh                  # variant: background-probability calibration sweep
    ├── pilot_s4.sh                    # variant: stride 4
    ├── aggregate.py                   # aggregates per-video results into a summary
```

(The canonical, runnable copies of `eval_long_video.py` and
`split_long_video.py` live in `working_directory/` and `tools/`; these shell
scripts are the as-run records from server2 and contain server-absolute paths.)

`scripts/` holds **copies**; the originals still sit in
`work_dirs/long_eval_18cls/` and `work_dirs/base18_eval/` on server2 and are
unchanged, so existing workflows keep working.

### Reproducing the clip-level numbers

```bash
cd ~/my_mmaction2

# 1. run the test set once and dump per-clip scores (4 GPUs)
bash evaluation/scripts/base18_test.sh

# 2. detection rate + classification accuracy  ->  eval_det_cls.txt
python working_directory/eval_det_cls.py \
    --preds work_dirs/base18_eval/preds.pkl \
    --ann data/myvideo/myvideo_val_list.txt \
    --labels data/myvideo/classInd.txt \
    --bg-index 0

# 3. the three clip-level figures
python working_directory/plot_det_cls.py \
    --preds work_dirs/base18_eval/preds.pkl \
    --ann data/myvideo/myvideo_val_list.txt \
    --labels data/myvideo/classInd.txt \
    --bg-index 0 --out-dir work_dirs/base18_eval
```

### Reproducing the long-video numbers

```bash
cd ~/my_mmaction2
# 1. sliding-window inference over the 63 test recordings, 4 GPUs in parallel
for g in 0 1 2 3; do bash evaluation/scripts/run_fleet.sh $g & done; wait   # ~45 min
# 2. per-video evaluation against the Excel annotations
bash evaluation/scripts/eval_all.sh
# 3. background-probability calibration sweep (alpha = 0.5 / 0.25 / 0.1)
bash evaluation/scripts/sweep_supp.sh
# 4. aggregate
cd work_dirs/long_eval_18cls
python ~/my_mmaction2/evaluation/scripts/aggregate.py eval_supp0.1 --exclude-label Walking
```

Step 4 reproduces the reported **9.3%** duration error. Note the two arguments:

* `eval_supp0.1` — the **deployment configuration**, in which the `Stand`
  (background) probability is scaled by 0.1 at inference. This is the single
  most important post-processing step: it takes the mean duration error from
  29.1% down to 9.3% and lifts frame-level mIoU from 0.549 to 0.608. Temporal
  smoothing (`eval_sm3`) has almost no effect by comparison.
* `--exclude-label Walking` — `Walking` is background by design in the deployed
  system and is therefore not an activity, matching the report's vocabulary.

| Variant | mean rel. error (all) | mean rel. error (excl. `Walking`) | median |
|---|---|---|---|
| `eval` (no calibration) | 30.9% | 29.1% | 15.5% |
| **`eval_supp0.1`** | 11.2% | **9.3%** | 0.0% |
| `eval_supp0.25` | 15.3% | 13.3% | 0.0% |
| `eval_supp0.5` | 22.2% | 20.3% | 3.2% |
| `eval_sm3` (smoothing) | 31.4% | 29.5% | 15.4% |

> The `aggregate.py` in `evaluation/scripts/` takes the evaluation directory as
> an argument. The original in `work_dirs/long_eval_18cls/` has `eval/`
> hard-coded, which is why the summary checked in there shows the 30.9%
> baseline rather than the 9.3% figure quoted in the report. Both numbers are
> correct — they are different configurations. The baseline copy is kept as
> `results/aggregate_summary_baseline.txt` for comparison.

The long-video evaluation additionally needs the raw session recordings
(`raw_long/`, ~13 GB — on server2 at `~/action/data/raw_long/`, integrity
manifest in `data/raw_long_sha256.txt`). The annotation spreadsheets and
`frame_counts.json` are already tracked in this repository under
`data/excel/`.

---

## Known issues

* **Hard-coded paths in configs.** The 4-class fine-tune configs set
  `load_from` to absolute checkpoint paths under `/home/zhuyusi/my_mmaction2/`.
  They work on server2 as-is; adjust them for any other machine. (The demo
  scripts formerly had absolute paths too; those are now repo-relative.)
* **`num_classes=4` in the shared base config** — see the warning under
  *Training*.
* **Several near-identical demo scripts** (`long_video_demo_4cls.py`,
  `long_video_demo_4cls_v2.py`, `long_video_4cls.py`, plus `.backup_*` copies).
  `long_video_demo_4cls_v2.py` is the current one.

## Acknowledgements

- **Yusi Zhu** — training of the general 17-action-class (+ background) base
  model and curation of the trained models.
- Repository restructuring, the evaluation pipeline (detection rate,
  classification accuracy, activity duration error) and documentation were
  prepared with the assistance of **Claude Fable 5** (Anthropic).
- Built on [MMAction2](https://github.com/open-mmlab/mmaction2) (Apache
  License 2.0) by OpenMMLab. The `mmaction/`, `tools/`, `tests/` and upstream
  `configs/` trees derive from it; see `LICENSE` for the upstream terms.
