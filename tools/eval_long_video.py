#!/usr/bin/env python3
"""Evaluate long-video action recognition with sliding-window predictions.

Cleaned-up CLI version of the analysis code from:
  /home/zhuyusi/action/data/analyze_bg.ipynb
  /home/zhuyusi/action/data/analyze_demo.ipynb

Outputs:
  - Frame-level mIoU and per-class IoU
  - Activity duration error (% of segments with duration error < threshold)
  - Predicted / True label-over-time visualization (PNG)
  - Optional CSV/JSON results

Usage:
  python eval_long_video.py \\
    --pred-pkl test/results_10_15_bg_included_7289.pkl \\
    --video-name 11_boss_C.mp4 \\
    --excel-dir data/excel \\
    --scene boss \\
    --class-ind data/myvideo/classInd.txt \\
    --frame-counts data/raw_long/frame_counts.json \\
    --window-size 64 --stride 16 \\
    --threshold 0.0 --fps 30 \\
    --out-dir results/eval_11_boss_C
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mode as scipy_mode

try:
    import torch
except ImportError:
    torch = None


# ============================================================
# Constants
# ============================================================

SCENE_SHEET_ALIASES = {
    "boss": ["Boss Fight", "boss", "Boss"],
    "bowling": ["Bowling", "bowling"],
    "candy": ["Candy", "candy"],
    "gallery": ["Gallery", "gallery"],
    "museum": ["Gaming Museum", "Museum", "museum", "gaming museum"],
    "travel": ["Travel", "travel"],
}

COMMON_ROWID_TO_LABEL = {
    1: "Walking", 2: "Walking",
    3: "Running", 4: "Running", 5: "Running",
    6: "Jumping", 7: "Jumping", 8: "Jumping",
    9: "Bending_Down", 10: "Bending_Down", 11: "Bending_Down",
    12: "Stand", 13: "Stand", 14: "Stand",
    15: "Squatting", 16: "Squatting", 17: "Squatting",
    18: "Raising_hand", 19: "Raising_hand", 20: "Raising_hand",
}

SCENE_ROWID_TO_LABEL = {
    "boss": {21: "Waive", 22: "Throw", 23: "Waive_sword", 24: "Cutting", 25: "Shooting"},
    "bowling": {21: "Bowling"},
    "candy": {21: "Shooting"},
    "gallery": {21: "Move_Controller", 22: "Waive_sword", 23: "Measure_Length"},
    "museum": {21: "Move_Controller", 22: "Picking_item", 23: "Measure_Length"},
    "travel": {21: "Move_Controller", 22: "Catching_fish", 23: "Grabbing", 24: "Measure_Length"},
}


# ============================================================
# Label mapping
# ============================================================

def load_label_mapping(path: str | Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    label2id: Dict[str, int] = {}
    id2label: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            idx = int(parts[0])
            label = parts[1]
            label2id[label] = idx
            id2label[idx] = label
    return label2id, id2label


# ============================================================
# Load predictions
# ============================================================

def _to_numpy(x: Any) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_window_scores_from_pkl(path: str | Path) -> np.ndarray:
    """Load pred_score from MMAction2 --dump pkl.

    Returns: [num_windows, num_classes]
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list in pkl, got {type(data)}")

    scores = []
    for item in data:
        if isinstance(item, dict) and "pred_score" in item:
            arr = _to_numpy(item["pred_score"]).astype(np.float32).reshape(-1)
            scores.append(arr)
        else:
            raise ValueError("pkl items must be dicts with 'pred_score' key")

    return np.stack(scores, axis=0)


# ============================================================
# Sliding-window fusion
# ============================================================

def windows_to_frame_labels(
    pred_score: np.ndarray,
    total_frames: int,
    window_size: int,
    stride: int,
    threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse overlapping sliding-window probabilities to per-frame labels.

    Args:
        pred_score: [num_windows, num_classes]
        total_frames: total number of frames in the video
        window_size: number of frames each window covers
        stride: step between consecutive windows
        threshold: if background (class 0) has highest prob AND > threshold,
                   assign background; otherwise assign best non-background class.

    Returns:
        frame_labels: [total_frames] int array
        frame_prob_mean: [total_frames, num_classes]
        frame_count: [total_frames] how many windows cover each frame
    """
    num_windows, num_classes = pred_score.shape
    frame_prob_sum = np.zeros((total_frames, num_classes), dtype=np.float64)
    frame_count = np.zeros(total_frames, dtype=np.int32)

    for i in range(num_windows):
        start = i * stride
        if start >= total_frames:
            break
        end = min(start + window_size, total_frames)
        frame_prob_sum[start:end] += pred_score[i]
        frame_count[start:end] += 1

    # Average probabilities
    frame_prob_mean = np.zeros_like(frame_prob_sum)
    covered = frame_count > 0
    frame_prob_mean[covered] = frame_prob_sum[covered] / frame_count[covered, None]

    # Decide per-frame label
    frame_labels = np.zeros(total_frames, dtype=np.int32)
    for idx in range(total_frames):
        if frame_count[idx] == 0:
            frame_labels[idx] = 0
            continue
        probs = frame_prob_mean[idx]
        max_label = int(probs.argmax())
        max_prob = probs[max_label]

        if max_label == 0 and max_prob > threshold:
            frame_labels[idx] = 0
        else:
            # Best non-background class
            if num_classes > 1:
                nonzero_probs = probs[1:]
                frame_labels[idx] = int(nonzero_probs.argmax()) + 1
            else:
                frame_labels[idx] = 0

    return frame_labels, frame_prob_mean, frame_count


# ============================================================
# Ground-truth from Excel
# ============================================================

def build_rowid_to_label(scene: str) -> Dict[int, str]:
    mapping = dict(COMMON_ROWID_TO_LABEL)
    mapping.update(SCENE_ROWID_TO_LABEL.get(scene.lower(), {}))
    return mapping


def resolve_excel_path(excel_dir: str | Path, prefix: int) -> Path:
    candidates = [
        Path(excel_dir) / f"DataCollection_{prefix:02d}.xlsx",
        Path(excel_dir) / f"DataCollection_{prefix}.xlsx",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No DataCollection Excel for prefix={prefix} in {excel_dir}")


def resolve_sheet_name(xlsx_path: str | Path, scene: str) -> str:
    xls = pd.ExcelFile(xlsx_path)
    sheets = list(xls.sheet_names)
    aliases = SCENE_SHEET_ALIASES.get(scene.lower(), [scene])
    normalized = {s.lower().strip(): s for s in sheets}
    for alias in aliases:
        key = alias.lower().strip()
        if key in normalized:
            return normalized[key]
    # Fuzzy fallback
    for sheet in sheets:
        if scene.lower() in sheet.lower():
            return sheet
    raise ValueError(f"Cannot resolve sheet for scene={scene!r}. Available: {sheets}")


def read_gt_intervals(
    xlsx_path: str | Path,
    sheet_name: str,
    rowid2label: Mapping[int, str],
    start_col_name: str = "Repetition 1 Start",
) -> Dict[str, List[Tuple[int, int]]]:
    """Read ground-truth action intervals from Excel."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if start_col_name not in df.columns:
        # Try to find a column containing "Repetition" and "Start"
        for col in df.columns:
            if "repetition" in str(col).lower() and "start" in str(col).lower():
                start_col_name = col
                break
        else:
            raise ValueError(
                f"Start column not found in {xlsx_path}:{sheet_name}. "
                f"Columns: {list(df.columns)}"
            )

    start_col_idx = df.columns.get_loc(start_col_name)
    action_dict: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    for idx, row in df.iterrows():
        row_number = idx + 1
        label = rowid2label.get(row_number)
        if label is None:
            continue

        row_values = row.iloc[start_col_idx:]
        non_null_positions = [i for i, v in enumerate(row_values) if pd.notnull(v)]

        for i in range(0, len(non_null_positions) - 1, 2):
            try:
                start_frame = int(row_values.iloc[non_null_positions[i]])
                end_frame = int(row_values.iloc[non_null_positions[i + 1]])
            except (ValueError, TypeError):
                continue
            if end_frame >= start_frame:
                action_dict[label].append((start_frame, end_frame))

    return dict(action_dict)


def intervals_to_frame_labels(
    intervals_by_label: Mapping[str, Sequence[Tuple[int, int]]],
    total_frames: int,
    label2id: Mapping[str, int],
    unlabeled_value: int = -1,
) -> np.ndarray:
    """Convert action intervals to per-frame label array."""
    labels = np.full(total_frames, unlabeled_value, dtype=np.int32)
    for label, intervals in intervals_by_label.items():
        if label not in label2id:
            print(f"  Warning: '{label}' not in label mapping, skipped.")
            continue
        label_id = int(label2id[label])
        for start_frame, end_frame in intervals:
            s = max(0, int(start_frame) - 1)
            e = min(total_frames - 1, int(end_frame) - 1)
            if s <= e:
                labels[s:e + 1] = label_id
    return labels


# ============================================================
# Metrics
# ============================================================

def compute_miou(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    id2label: Dict[int, str],
    ignore_label: int = -1,
) -> Tuple[float, Dict[str, float]]:
    """Compute mean IoU over all classes present in true_labels."""
    mask = true_labels != ignore_label
    pred = pred_labels[mask]
    true = true_labels[mask]

    classes = np.unique(true)
    iou_dict: Dict[str, float] = {}

    for c in classes:
        pred_c = pred == c
        true_c = true == c
        intersection = np.logical_and(pred_c, true_c).sum()
        union = np.logical_or(pred_c, true_c).sum()
        iou = float(intersection / union) if union > 0 else float("nan")
        label_name = id2label.get(int(c), f"class_{c}")
        iou_dict[label_name] = iou

    valid = [v for v in iou_dict.values() if not math.isnan(v)]
    mean_iou = float(np.mean(valid)) if valid else 0.0
    return mean_iou, iou_dict


def compute_duration_error(
    pred_labels: np.ndarray,
    gt_intervals: Mapping[str, Sequence[Tuple[int, int]]],
    label2id: Mapping[str, int],
    error_threshold: float = 0.1,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Compute activity duration error metric.

    For each GT segment, find the predicted segment(s) that overlap with it,
    compute the predicted duration as the total frames predicted as that class
    within the GT interval, then compute relative duration error.

    Returns:
        ratio: fraction of GT segments with relative duration error < threshold
        details: list of per-segment results
    """
    details: List[Dict[str, Any]] = []
    total_segments = 0
    within_threshold = 0

    for label, intervals in gt_intervals.items():
        if label not in label2id:
            continue
        label_id = label2id[label]
        for start_frame, end_frame in intervals:
            s = max(0, int(start_frame) - 1)
            e = min(len(pred_labels) - 1, int(end_frame) - 1)
            if s > e:
                continue
            gt_duration = e - s + 1
            pred_duration = int((pred_labels[s:e + 1] == label_id).sum())
            if gt_duration == 0:
                continue
            rel_error = abs(pred_duration - gt_duration) / gt_duration
            is_within = rel_error < error_threshold
            total_segments += 1
            if is_within:
                within_threshold += 1
            details.append({
                "label": label,
                "gt_start": int(start_frame),
                "gt_end": int(end_frame),
                "gt_duration": gt_duration,
                "pred_duration": pred_duration,
                "rel_error": round(rel_error, 4),
                "within_threshold": is_within,
            })

    ratio = within_threshold / total_segments if total_segments > 0 else 0.0
    return ratio, details


# ============================================================
# Visualization
# ============================================================

def plot_labels_over_time(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    id2label: Dict[int, str],
    fps: int = 30,
    out_path: Optional[str | Path] = None,
    title_prefix: str = "",
    ignore_label: int = -1,
):
    """Plot true and predicted labels over time (seconds)."""
    n_frames = min(len(true_labels), len(pred_labels))
    n_seconds = n_frames // fps
    if n_seconds < 1:
        print("  Warning: video too short for time plot.")
        return

    true_trimmed = true_labels[:n_seconds * fps].reshape(n_seconds, fps)
    pred_trimmed = pred_labels[:n_seconds * fps].reshape(n_seconds, fps)

    # Per-second mode
    true_per_sec = scipy_mode(true_trimmed, axis=1, keepdims=False).mode.flatten()
    pred_per_sec = scipy_mode(pred_trimmed, axis=1, keepdims=False).mode.flatten()

    seconds = np.arange(n_seconds)

    # Collect all unique labels for consistent y-axis
    all_labels = sorted(set(np.unique(true_per_sec).tolist() + np.unique(pred_per_sec).tolist()))
    all_labels = [l for l in all_labels if l != ignore_label]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(16, n_seconds / 20), 8), sharex=True)

    # True labels
    colors_true = true_per_sec.astype(float)
    ax1.scatter(seconds, true_per_sec, c=colors_true, cmap="tab20", s=8, marker="|")
    ax1.set_yticks(all_labels)
    ax1.set_yticklabels([id2label.get(l, f"cls_{l}") for l in all_labels], fontsize=8)
    ax1.set_title(f"{title_prefix}True Labels over Time")
    ax1.set_ylabel("Action")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # Predicted labels
    colors_pred = pred_per_sec.astype(float)
    ax2.scatter(seconds, pred_per_sec, c=colors_pred, cmap="tab20", s=8, marker="|")
    ax2.set_yticks(all_labels)
    ax2.set_yticklabels([id2label.get(l, f"cls_{l}") for l in all_labels], fontsize=8)
    ax2.set_title(f"{title_prefix}Predicted Labels over Time")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Action")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {out_path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate long-video action recognition (sliding-window mIoU + duration error)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pred-pkl", required=True, help="Path to MMAction2 --dump pkl file")
    p.add_argument("--video-name", required=True, help="Video filename, e.g. 11_boss_C.mp4")
    p.add_argument("--excel-dir", required=True, help="Directory containing DataCollection_XX.xlsx")
    p.add_argument("--scene", default=None, help="Scene name (auto-detected from video-name if omitted)")
    p.add_argument("--class-ind", required=True, help="Path to classInd.txt")
    p.add_argument("--frame-counts", default=None, help="Path to frame_counts.json (auto total_frames)")
    p.add_argument("--total-frames", type=int, default=None, help="Override total frame count")
    p.add_argument("--window-size", type=int, default=64, help="Sliding window size in frames")
    p.add_argument("--stride", type=int, default=16, help="Sliding window stride in frames")
    p.add_argument("--threshold", type=float, default=0.0, help="Background confidence threshold")
    p.add_argument("--fps", type=int, default=30, help="Video FPS for time-axis plots")
    p.add_argument("--duration-error-threshold", type=float, default=0.1,
                   help="Relative duration error threshold (default 0.1 = 10%%)")
    p.add_argument("--ignore-unlabeled", action="store_true",
                   help="Ignore unlabeled frames (GT=-1) in mIoU computation")
    p.add_argument("--out-dir", default="eval_output", help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load label mapping ---
    print(f"Loading class mapping: {args.class_ind}")
    label2id, id2label = load_label_mapping(args.class_ind)
    print(f"  {len(label2id)} classes loaded.")

    # --- Determine scene and prefix ---
    scene = args.scene or infer_scene_from_video_name(args.video_name)
    if scene is None:
        raise ValueError(f"Cannot infer scene from '{args.video_name}'. Use --scene explicitly.")
    prefix = infer_prefix_from_video_name(args.video_name)
    if prefix is None:
        raise ValueError(f"Cannot infer prefix (person id) from '{args.video_name}'.")
    print(f"  Video: {args.video_name} | Scene: {scene} | Prefix: {prefix}")

    # --- Determine total frames ---
    total_frames = args.total_frames
    if total_frames is None and args.frame_counts:
        fc = json.load(open(args.frame_counts))
        key = args.video_name
        if key not in fc:
            # Try without extension
            key_noext = Path(key).stem
            for k in fc:
                if Path(k).stem == key_noext:
                    key = k
                    break
        if key in fc:
            total_frames = int(fc[key])
        else:
            raise ValueError(f"'{args.video_name}' not found in frame_counts.json")
    if total_frames is None:
        raise ValueError("Must provide --total-frames or --frame-counts")
    print(f"  Total frames: {total_frames}")

    # --- Load predictions ---
    print(f"Loading predictions: {args.pred_pkl}")
    pred_score = load_window_scores_from_pkl(args.pred_pkl)
    print(f"  {pred_score.shape[0]} windows, {pred_score.shape[1]} classes")

    # --- Sliding-window fusion ---
    print(f"Fusing windows: window_size={args.window_size}, stride={args.stride}, threshold={args.threshold}")
    pred_labels, frame_prob_mean, frame_count = windows_to_frame_labels(
        pred_score, total_frames, args.window_size, args.stride, args.threshold
    )
    uncovered = int((frame_count == 0).sum())
    if uncovered > 0:
        print(f"  Warning: {uncovered} frames not covered by any window.")

    # --- Load GT from Excel ---
    print(f"Loading GT from Excel (prefix={prefix}, scene={scene})")
    xlsx_path = resolve_excel_path(args.excel_dir, prefix)
    sheet_name = resolve_sheet_name(xlsx_path, scene)
    rowid2label = build_rowid_to_label(scene)
    gt_intervals = read_gt_intervals(xlsx_path, sheet_name, rowid2label)
    print(f"  Sheet: {sheet_name} | {sum(len(v) for v in gt_intervals.values())} GT segments")

    gt_frame_labels = intervals_to_frame_labels(gt_intervals, total_frames, label2id)

    # --- Compute mIoU ---
    ignore_val = -1 if args.ignore_unlabeled else None
    if args.ignore_unlabeled:
        # Only evaluate on labeled frames
        miou, per_class_iou = compute_miou(pred_labels, gt_frame_labels, id2label, ignore_label=-1)
    else:
        # Treat unlabeled as background (0)
        gt_for_eval = gt_frame_labels.copy()
        gt_for_eval[gt_for_eval == -1] = 0
        miou, per_class_iou = compute_miou(pred_labels, gt_for_eval, id2label, ignore_label=-1)

    print(f"\n{'='*50}")
    print(f"  Mean IoU: {miou:.4f}")
    print(f"  Per-class IoU:")
    for label_name, iou_val in sorted(per_class_iou.items(), key=lambda x: -x[1]):
        print(f"    {label_name:20s}: {iou_val:.4f}")

    # --- Compute duration error ---
    dur_ratio, dur_details = compute_duration_error(
        pred_labels, gt_intervals, label2id, args.duration_error_threshold
    )
    print(f"\n  Duration Error < {args.duration_error_threshold*100:.0f}%: "
          f"{dur_ratio*100:.1f}% of segments ({sum(1 for d in dur_details if d['within_threshold'])}/{len(dur_details)})")
    print(f"{'='*50}\n")

    # --- Plot ---
    gt_for_plot = gt_frame_labels.copy()
    gt_for_plot[gt_for_plot == -1] = 0  # Show unlabeled as background in plot

    plot_path = out_dir / "labels_over_time.png"
    plot_labels_over_time(
        gt_for_plot, pred_labels, id2label,
        fps=args.fps, out_path=plot_path,
        title_prefix=f"{args.video_name} | ",
    )

    # --- Save results ---
    results = {
        "video_name": args.video_name,
        "scene": scene,
        "prefix": prefix,
        "total_frames": total_frames,
        "num_windows": int(pred_score.shape[0]),
        "window_size": args.window_size,
        "stride": args.stride,
        "threshold": args.threshold,
        "fps": args.fps,
        "mean_iou": round(miou, 4),
        "per_class_iou": {k: round(v, 4) for k, v in per_class_iou.items()},
        "duration_error_threshold": args.duration_error_threshold,
        "duration_error_ratio": round(dur_ratio, 4),
        "duration_error_details": dur_details,
    }

    json_path = out_dir / "eval_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Results JSON: {json_path}")

    # Save frame-level predictions as CSV for further analysis
    csv_path = out_dir / "frame_labels.csv"
    df = pd.DataFrame({
        "frame": np.arange(total_frames),
        "pred_label": pred_labels,
        "gt_label": gt_frame_labels,
        "pred_name": [id2label.get(int(l), "?") for l in pred_labels],
        "gt_name": [id2label.get(int(l), "unlabeled") if l != -1 else "unlabeled" for l in gt_frame_labels],
    })
    df.to_csv(csv_path, index=False)
    print(f"  Frame labels CSV: {csv_path}")

    print("\nDone.")


def infer_scene_from_video_name(video_name: str) -> Optional[str]:
    lower = video_name.lower()
    for scene in SCENE_ROWID_TO_LABEL:
        if scene in lower:
            return scene
    return None


def infer_prefix_from_video_name(video_name: str) -> Optional[int]:
    match = re.match(r"(\d+)[_\-]", Path(video_name).name)
    if match:
        return int(match.group(1))
    return None


if __name__ == "__main__":
    main()
