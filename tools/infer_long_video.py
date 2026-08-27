#!/usr/bin/env python
"""Sliding-window inference over a long video, dumping per-window scores to pkl.

Window semantics match tools/eval_long_video.py exactly:
  window i covers raw frames [i*stride, i*stride + window_size), sampled every
  input_step frames -> clip_len frames fed to the model.

Preprocessing follows the single-view val pipeline of my_swin.py
(Resize short side 256 -> CenterCrop 224), same as deployed single-view
inference. Sequential decode with a rolling buffer (decode each frame once).

Output pkl: list of {'pred_score': np.ndarray[num_classes]} - loadable by
eval_long_video.py::load_window_scores_from_pkl.
"""
import argparse
import pickle
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.config import Config

from mmaction.apis import init_recognizer
from mmaction.structures import ActionDataSample


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--num-classes', type=int, default=18)
    p.add_argument('--window-size', type=int, default=64)
    p.add_argument('--stride', type=int, default=16)
    p.add_argument('--input-step', type=int, default=2)
    p.add_argument('--batch-windows', type=int, default=8)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-windows', type=int, default=0, help='debug: limit')
    p.add_argument('--bg-scale', type=float, default=1.0,
                   help='scale background prob before saving; 1.0 = raw. '
                        '18-class base model only.')
    p.add_argument('--bg-index', type=int, default=0)
    p.add_argument('--out', required=True)
    return p.parse_args()


def preprocess_frame(f_bgr):
    """BGR frame -> RGB 224x224 uint8 (short side 256 + center crop)."""
    h, w = f_bgr.shape[:2]
    scale = 256.0 / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    f = cv2.resize(f_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    y0, x0 = (nh - 224) // 2, (nw - 224) // 2
    f = f[y0:y0 + 224, x0:x0 + 224]
    return np.ascontiguousarray(f[:, :, ::-1])  # BGR -> RGB


def flush(model, pending, results):
    if not pending:
        return
    data = {
        'inputs': list(pending),
        'data_samples': [ActionDataSample() for _ in pending],
    }
    with torch.no_grad():
        preds = model.test_step(data)
    for p in preds:
        s = p.pred_score.cpu().numpy().astype(np.float32)
        if ARGS.bg_scale != 1.0:
            s = s.copy()
            s[ARGS.bg_index] *= ARGS.bg_scale
            s = s / s.sum()
        results.append({'pred_score': s})
    pending.clear()


def main():
    global ARGS
    args = parse_args()
    ARGS = args
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict({'model.cls_head.num_classes': args.num_classes})
    model = init_recognizer(cfg, args.checkpoint, device=args.device)
    model.eval()

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_windows = max(0, (total - args.window_size) // args.stride + 1)
    if args.max_windows:
        n_windows = min(n_windows, args.max_windows)
    print(f'{args.video}: {total} frames -> {n_windows} windows', flush=True)

    buf = deque(maxlen=args.window_size)
    results = []
    pending = []
    next_start = 0
    frame_idx = 0
    last_print = 0
    while len(results) + len(pending) < n_windows:
        ret, frame = cap.read()
        if not ret:
            break
        buf.append(preprocess_frame(frame))
        frame_idx += 1
        if frame_idx == next_start + args.window_size:
            arr = np.stack(list(buf)[::args.input_step])
            t = torch.from_numpy(arr).permute(3, 0, 1, 2)
            pending.append(t.unsqueeze(0).float())
            next_start += args.stride
            if len(pending) >= args.batch_windows:
                flush(model, pending, results)
                if len(results) - last_print >= 160:
                    last_print = len(results)
                    print(f'  {len(results)}/{n_windows}', flush=True)
    flush(model, pending, results)
    cap.release()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(results, f)
    print(f'saved {len(results)} windows -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
