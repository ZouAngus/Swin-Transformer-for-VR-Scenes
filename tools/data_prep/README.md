# Data preparation — how the evaluation data was produced

Pipeline from raw recordings to the data used in EVAL_detection_classification.md.

## 1. Raw recordings
15 participants x 6 scenes x 2 fixed camera views (C = centre, L = left),
full training sessions at 30 fps. Persons 11-15 (the test split) are archived
on this server at `~/action/data/raw_long/{11..15}/` (63 videos, ~13 GB).
Persons 1-10 (training split) live on server1: `/data/sda1/shared/`.

## 2. Annotation
One Excel workbook per person: `data/excel/DataCollection_XX.xlsx`
(copied here from `~/action/data/excel/`), one sheet per scene. Each row is an
action type (fixed row-id -> label mapping, common rows 1-20 + scene-specific
rows 21+; see SCENE_ROWID_TO_LABEL in `tools/eval_long_video.py`), and each
annotated repetition stores start/end frame indices. Frames are annotated on
view C; view L shares the annotation (~6-frame offset, see repo readme).
`data/excel/frame_counts.json` records the frame count of every raw video.

## 3. Clip-level test set (data/myvideo)
- Annotated intervals were cut from the raw recordings into per-action clips
  (naming: `s{person}_a{action}_{scene}_{row}_rep{rep}_frames_{start}_{end}_{view}.mp4`);
  this first cut was historically performed on server1
  (`/data/sda1/shared/sliced_video`). It can now be reproduced from this repo
  with `slice_raw_actions.py` (raw videos + Excel -> per-class clips).
- Long background (`_bg`) stretches were further split into 100-frame windows:
  `slice_bg.py` (train) / `slice_bg_val.py` (val).
- `file_list.py` walks `videos_train/` / `videos_val/` and generates
  `myvideo_train_list.txt` / `myvideo_val_list.txt` (labels from classInd.txt).
- `h264.py` optionally re-encodes clips for preview in VSCode.
Result: 14k training clips (persons 1-10) and 7,289 test clips (persons 11-15).

## 4. Long-video evaluation data
No cutting: sliding-window inference (`tools/infer_long_video.py`) runs
directly on the raw session recordings of persons 11-15. Ground-truth event
intervals are parsed from the same Excel workbooks by
`tools/eval_long_video.py` (also archived here; original at
`~/action/tools/eval_long_video.py`). `split_long_video.py` is the legacy
alternative that physically cuts a long video into sliding-window clips for
`tools/test.py`.
