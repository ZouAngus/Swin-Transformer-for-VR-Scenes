#!/usr/bin/env python
"""Cut raw long session videos into per-action clips using the Excel annotations.

This is the FIRST stage of the data pipeline (raw -> clips). It reproduces the
historical slicing that originally happened on server1, so that a third party
holding only (a) the raw long videos and (b) this repository can rebuild the
training / test clip sets from scratch:

    1. data/prep/slice_raw_actions.py   raw videos + Excel -> per-class clips
    2. data/prep/slice_bg_val.py        window long Stand/_bg clips (100 frames)
    3. data/prep/file_list.py           clips -> myvideo_{train,val}_list.txt

Excel parsing (sheet aliases, row-id -> label mapping, repetition columns) is
imported from working_directory/eval_long_video.py so the ground truth used for slicing is
byte-identical to the one used by the event-level evaluation.

Output layout (one folder per class, mirroring data/myvideo/videos_*):

    <out-dir>/<Label>/<person>_<scene>_row<row?>_frames_<start>_<end>_<view>.mp4
    <out-dir>/Stand/<person>_<scene>_<start>_<end>_<view>_bg.mp4   (Stand = background)

Usage (rebuild the test split, persons 11-15):

    python data/prep/slice_raw_actions.py \
        --raw-dir /path/to/raw_long \
        --excel-dir data/excel \
        --out-dir data/myvideo/videos_val \
        --persons 11 12 13 14 15
"""
import argparse
import importlib.util
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2

# import Excel helpers from working_directory/eval_long_video.py without packaging
_ELV = Path(__file__).resolve().parents[2] / 'working_directory' / 'eval_long_video.py'
spec = importlib.util.spec_from_file_location('eval_long_video', _ELV)
elv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(elv)

SCENES = ['boss', 'bowling', 'candy', 'gallery', 'museum', 'travel']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', required=True,
                   help='folder containing <person>/<person>_<scene>_<view>.mp4')
    p.add_argument('--excel-dir', required=True, help='DataCollection_XX.xlsx folder')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--persons', type=int, nargs='+', required=True)
    p.add_argument('--views', nargs='+', default=['C', 'L'])
    p.add_argument('--scenes', nargs='+', default=SCENES)
    p.add_argument('--min-frames', type=int, default=16,
                   help='skip intervals shorter than this')
    return p.parse_args()


def cut_video(video_path, jobs, fps_out=None):
    """Sequentially decode video once; write every (start, end, out_path) job."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'  !! cannot open {video_path}')
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    jobs = sorted(jobs)                      # by start frame
    open_writers = []                        # (end, writer)
    ji = 0
    frame_idx = 0
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        while ji < len(jobs) and jobs[ji][0] == frame_idx:
            s, e, out_path = jobs[ji]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            open_writers.append([e, cv2.VideoWriter(str(out_path), fourcc,
                                                    fps_out or fps, (w, h))])
            ji += 1
        for ow in open_writers:
            ow[1].write(frame)
        for ow in [o for o in open_writers if o[0] <= frame_idx]:
            ow[1].release()
            open_writers.remove(ow)
            written += 1
        frame_idx += 1
        if ji >= len(jobs) and not open_writers:
            break
    for _, wtr in open_writers:              # intervals past EOF
        wtr.release()
        written += 1
    cap.release()
    return written


def main():
    args = parse_args()
    out_root = Path(args.out_dir)
    total = 0
    for person in args.persons:
        xlsx = elv.resolve_excel_path(args.excel_dir, person)
        for scene in args.scenes:
            try:
                sheet = elv.resolve_sheet_name(xlsx, scene)
                gt = elv.read_gt_intervals(xlsx, sheet, elv.build_rowid_to_label(scene))
            except Exception as exc:
                print(f'  !! {person}/{scene}: annotation error: {exc}')
                continue
            for view in args.views:
                video = Path(args.raw_dir) / str(person) / f'{person}_{scene}_{view}.mp4'
                if not video.exists():
                    print(f'  !! missing raw video: {video}')
                    continue
                jobs = []
                for label, intervals in gt.items():
                    for s, e in intervals:
                        if e - s + 1 < args.min_frames:
                            continue
                        if label == 'Stand':
                            name = f'{person}_{scene}_{s}_{e}_{view}_bg.mp4'
                        else:
                            name = f'{person}_{scene}_frames_{s}_{e}_{view}.mp4'
                        jobs.append((s, e, out_root / label / name))
                n = cut_video(video, jobs)
                total += n
                print(f'{person}/{scene}/{view}: {n} clips')
    print(f'done: {total} clips -> {out_root}')
    print('next: run slice_bg_val.py on the Stand/_bg clips, then file_list.py')


if __name__ == '__main__':
    main()
