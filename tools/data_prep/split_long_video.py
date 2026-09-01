#!/usr/bin/env python3
"""Split a long video into sliding-window clips using OpenCV (no ffmpeg needed).

Generates overlapping short clips and an annotation txt for MMAction2 test.py --dump.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Split long video into sliding-window clips (cv2)")
    parser.add_argument("--video", required=True, help="Path to long video")
    parser.add_argument("--frame-counts", default=None, help="frame_counts.json")
    parser.add_argument("--total-frames", type=int, default=None, help="Override total frames")
    parser.add_argument("--window-size", type=int, default=64, help="Window size in frames")
    parser.add_argument("--stride", type=int, default=16, help="Stride in frames")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = args.total_frames
    if total_frames is None and args.frame_counts:
        fc = json.load(open(args.frame_counts))
        key = video_path.name
        if key not in fc:
            for k in fc:
                if Path(k).stem == video_path.stem:
                    key = k
                    break
        if key in fc:
            total_frames = int(fc[key])
    if total_frames is None:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps:.1f}")
    print(f"Window: {args.window_size} frames, Stride: {args.stride} frames")

    out_dir = Path(args.out_dir)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Calculate windows
    windows = []
    start = 0
    while start + args.window_size <= total_frames:
        windows.append((start, start + args.window_size))
        start += args.stride
    if start < total_frames and (not windows or windows[-1][1] < total_frames):
        windows.append((start, min(start + args.window_size, total_frames)))

    print(f"Number of windows: {len(windows)}")

    # Read all frames into memory in one pass (more efficient than seeking)
    print("Reading video frames...")
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
        if len(all_frames) % 5000 == 0:
            print(f"  Read {len(all_frames)} frames...")
    cap.release()
    actual_frames = len(all_frames)
    print(f"  Read {actual_frames} frames total.")

    # Adjust total_frames if needed
    if actual_frames < total_frames:
        print(f"  Warning: actual frames ({actual_frames}) < expected ({total_frames}), adjusting.")
        total_frames = actual_frames
        windows = []
        start = 0
        while start + args.window_size <= total_frames:
            windows.append((start, start + args.window_size))
            start += args.stride
        if start < total_frames and (not windows or windows[-1][1] < total_frames):
            windows.append((start, min(start + args.window_size, total_frames)))
        print(f"  Adjusted windows: {len(windows)}")

    # Write clips
    h, w = all_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    ann_lines = []
    for i, (ws, we) in enumerate(windows):
        clip_name = f"clip_{i:05d}_{ws}_{we}.mp4"
        clip_path = clips_dir / clip_name

        if not clip_path.exists():
            writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (w, h))
            for fi in range(ws, min(we, actual_frames)):
                writer.write(all_frames[fi])
            writer.release()

        ann_lines.append(f"clips/{clip_name} 0")

        if (i + 1) % 200 == 0:
            print(f"  Written {i+1}/{len(windows)} clips...")

    # Write annotation
    ann_path = out_dir / "sliding_window_ann.txt"
    with open(ann_path, "w") as f:
        f.write("\n".join(ann_lines) + "\n")

    print(f"\nDone! {len(windows)} clips extracted.")
    print(f"Annotation: {ann_path}")
    print(f"Clips: {clips_dir}")


if __name__ == "__main__":
    main()
