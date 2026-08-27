#!/usr/bin/env python
"""Long-video demo for the 4-class Swin-tiny recognizer.

Sliding window inference with:
  - per-window softmax probabilities
  - moving-average temporal smoothing (window K)
  - confidence threshold (tau) for non-Stand classes
  - hysteresis (require N consecutive frames to switch state)
  - overlay rendering with current label + smoothed prob
  - JSON dump of per-window probs and consolidated segments

Run on server2 inside the mmaction2 conda env.
"""
import argparse
import json
import os
from collections import deque

import cv2
import numpy as np
import torch
from mmengine import Config
from mmengine.dataset import Compose

from mmaction.apis import init_recognizer

FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--video', required=True)
    p.add_argument('--label', required=True, help='label txt, one per line')
    p.add_argument('--out-video', default=None, help='annotated mp4 (optional)')
    p.add_argument('--out-json', required=True, help='per-window + segments json')
    p.add_argument('--stride', type=float, default=0.25,
                   help='prediction stride as fraction of sample_length')
    p.add_argument('--smooth-k', type=int, default=5,
                   help='moving-average window over softmax probs')
    p.add_argument('--threshold', type=float, default=0.6,
                   help='min smoothed prob for non-Stand classes')
    p.add_argument('--hysteresis', type=int, default=3,
                   help='consecutive predictions required to switch state')
    p.add_argument('--stand-index', type=int, default=0,
                   help='label index for "Stand" / default state')
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


def build_test_pipeline(cfg):
    test_pipeline = cfg.test_dataloader.dataset.pipeline
    # Drop decode/init steps; we feed numpy frames directly.
    keep = []
    for step in test_pipeline:
        t = step['type']
        if t in ('DecordInit', 'DecordDecode', 'OpenCVInit', 'OpenCVDecode',
                 'PyAVInit', 'PyAVDecode', 'RawFrameDecode', 'SampleFrames'):
            continue
        keep.append(step)
    return Compose(keep)


def sample_length_from_cfg(cfg):
    for step in cfg.test_dataloader.dataset.pipeline:
        if step['type'] == 'SampleFrames':
            return step['clip_len'] * step['frame_interval'], step['clip_len']
    raise RuntimeError('SampleFrames not found in test pipeline')


@torch.no_grad()
def infer_window(model, pipeline, frames, clip_len):
    # Uniformly subsample clip_len frames from the window.
    idx = np.linspace(0, len(frames) - 1, clip_len).astype(int)
    clip = [frames[i] for i in idx]
    data = dict(
        imgs=clip,
        num_clips=1,
        clip_len=clip_len,
        modality='RGB',
        total_frames=len(clip),
        frame_inds=np.arange(clip_len),
        start_index=0,
    )
    data = pipeline(data)
    inputs = data['inputs']
    # data_preprocessor expects inputs as a list (per-sample tensors).
    if isinstance(inputs, list):
        inputs = [t.to(next(model.parameters()).device) for t in inputs]
    else:
        inputs = [inputs.to(next(model.parameters()).device)]
    data_samples = [data['data_samples']]
    out = model.test_step(dict(inputs=inputs, data_samples=data_samples))
    pred = out[0]
    scores = pred.pred_score.detach().cpu().numpy()
    return scores  # shape (num_classes,)


def consolidate_segments(per_window, fps, labels, stand_idx):
    """Turn per-window decisions into [{label,start_s,end_s,avg_prob}] segments."""
    segs = []
    cur = None
    for w in per_window:
        lab = w['decision']
        if cur is None or lab != cur['label']:
            if cur is not None:
                segs.append(cur)
            cur = dict(label=lab, start_s=w['center_s'],
                       end_s=w['center_s'], probs=[w['prob']])
        else:
            cur['end_s'] = w['center_s']
            cur['probs'].append(w['prob'])
    if cur is not None:
        segs.append(cur)
    for s in segs:
        s['avg_prob'] = float(np.mean(s['probs']))
        del s['probs']
    return segs


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    device = torch.device(args.device)
    model = init_recognizer(cfg, args.checkpoint, device=device)
    model.eval()
    pipeline = build_test_pipeline(cfg)
    sample_length, clip_len = sample_length_from_cfg(cfg)

    with open(args.label) as f:
        labels = [l.strip() for l in f if l.strip()]
    assert len(labels) == cfg.model.cls_head.num_classes, \
        f'label count {len(labels)} != num_classes {cfg.model.cls_head.num_classes}'

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.out_video, fourcc, fps, (w, h))

    stride = max(1, int(round(sample_length * args.stride)))
    window = deque(maxlen=sample_length)
    prob_hist = deque(maxlen=args.smooth_k)
    pending_label = args.stand_index
    pending_count = 0
    current_label = args.stand_index
    current_prob = 0.0

    per_window = []
    frame_idx = 0
    next_pred_at = sample_length  # first prediction once we have a full window

    print(f'[info] frames={n_frames} fps={fps:.2f} sample_length={sample_length} '
          f'stride={stride} smooth_k={args.smooth_k} tau={args.threshold} '
          f'hyst={args.hysteresis}')

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        window.append(rgb)
        frame_idx += 1

        if len(window) == sample_length and frame_idx >= next_pred_at:
            scores = infer_window(model, pipeline, list(window), clip_len)
            prob_hist.append(scores)
            smoothed = np.mean(np.stack(prob_hist), axis=0)
            top_idx = int(np.argmax(smoothed))
            top_prob = float(smoothed[top_idx])

            # Threshold gating: non-stand needs prob >= tau, else fall back to stand.
            if top_idx != args.stand_index and top_prob < args.threshold:
                gated = args.stand_index
            else:
                gated = top_idx

            # Hysteresis: require N consecutive identical gated preds to switch.
            if gated == current_label:
                pending_label = gated
                pending_count = 0
            elif gated == pending_label:
                pending_count += 1
                if pending_count >= args.hysteresis:
                    current_label = gated
                    pending_count = 0
            else:
                pending_label = gated
                pending_count = 1

            current_prob = float(smoothed[current_label])
            center_s = (frame_idx - sample_length / 2) / fps
            per_window.append(dict(
                frame=frame_idx,
                center_s=round(center_s, 3),
                raw_top=labels[top_idx],
                raw_prob=round(top_prob, 4),
                decision=labels[current_label],
                prob=round(current_prob, 4),
                probs={labels[i]: round(float(smoothed[i]), 4)
                       for i in range(len(labels))},
            ))
            next_pred_at += stride

        if writer is not None:
            text = f'{labels[current_label]} {current_prob:.2f}'
            cv2.rectangle(frame, (0, 0), (380, 40), (0, 0, 0), -1)
            cv2.putText(frame, text, (10, 28), FONT, 0.9, (0, 255, 0), 2)
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()

    segments = consolidate_segments(per_window, fps, labels, args.stand_index)
    out = dict(
        video=args.video,
        fps=fps,
        n_frames=n_frames,
        sample_length=sample_length,
        stride=stride,
        smooth_k=args.smooth_k,
        threshold=args.threshold,
        hysteresis=args.hysteresis,
        labels=labels,
        per_window=per_window,
        segments=segments,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2)

    print(f'[done] windows={len(per_window)} segments={len(segments)}')
    for s in segments:
        if s['label'] != labels[args.stand_index]:
            print(f"  {s['label']:<14} {s['start_s']:6.2f}-{s['end_s']:6.2f}s "
                  f"avg_p={s['avg_prob']:.2f}")


if __name__ == '__main__':
    main()
