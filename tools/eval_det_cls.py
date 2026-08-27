#!/usr/bin/env python
"""Detection rate + classification accuracy from a tools/test.py preds.pkl dump.

Metric definitions (clip level, sliding-window clips):
  - detection rate:
        fraction of ground-truth ACTION clips (gt != background) that are
        NOT predicted as background, i.e. the model "detected" an action.
  - classification accuracy (among detected):
        of the detected action clips, the fraction whose predicted label
        equals the ground-truth label.
  - classification accuracy (all action clips):
        stricter variant, correct / all action clips (missed ones count
        as wrong).

Usage (18-class base model):
  python tools/eval_det_cls.py \
      --preds work_dirs/base18_eval/preds.pkl \
      --ann data/myvideo/myvideo_val_list.txt \
      --labels data/myvideo/classInd.txt \
      --bg-index 0

Works for the 4-class model too (labels file may be either
"<idx> <name>" like classInd.txt, or one name per line like label_4cls.txt).
"""
import argparse
import pickle

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--preds', required=True, help='pkl from tools/test.py --dump')
    p.add_argument('--ann', required=True, help='val_list.txt (path label)')
    p.add_argument('--labels', required=True,
                   help='classInd.txt ("idx name") or label file (one name per line)')
    p.add_argument('--bg-index', type=int, default=0,
                   help='index of the background class (default: 0 = Stand)')
    return p.parse_args()


def load_labels(path):
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1])
            else:
                labels.append(line)
    return labels


def main():
    args = parse_args()
    with open(args.preds, 'rb') as f:
        preds = pickle.load(f)
    with open(args.ann) as f:
        gts = [int(line.strip().rsplit(' ', 1)[1]) for line in f if line.strip()]
    labels = load_labels(args.labels)

    assert len(preds) == len(gts), f'{len(preds)} preds vs {len(gts)} gts'
    n = len(labels)
    bg = args.bg_index
    cm = np.zeros((n, n), dtype=int)
    for p, gt in zip(preds, gts):
        score = p['pred_score'] if isinstance(p, dict) else p.pred_score
        if hasattr(score, 'cpu'):
            score = score.cpu().numpy()
        score = np.asarray(score)
        cm[gt, int(score.argmax())] += 1

    total = cm.sum()
    print(f'samples={total}  top-1 acc={np.trace(cm) / total:.4f}  '
          f'(background class: {labels[bg]}, index {bg})')
    print()

    # ---- overall detection / classification metrics ----
    action = [i for i in range(n) if i != bg]
    n_action = cm[action, :].sum()                       # gt action clips
    n_missed = cm[action, bg].sum()                      # predicted as background
    n_detected = n_action - n_missed
    n_correct = sum(cm[i, i] for i in action)            # detected AND right label
    n_bg = cm[bg, :].sum()
    n_false_alarm = n_bg - cm[bg, bg]                    # background -> action

    det_rate = n_detected / n_action if n_action else 0.0
    cls_acc_detected = n_correct / n_detected if n_detected else 0.0
    cls_acc_all = n_correct / n_action if n_action else 0.0
    fa_rate = n_false_alarm / n_bg if n_bg else 0.0

    print(f'action clips (gt != bg):            {n_action}')
    print(f'  detected (pred != bg):            {n_detected}')
    print(f'  missed  (pred == bg):             {n_missed}')
    print(f'  correctly classified:             {n_correct}')
    print(f'background clips (gt == bg):        {n_bg}')
    print(f'  false alarms (pred != bg):        {n_false_alarm}')
    print()
    print(f'DETECTION RATE                    = {det_rate:.4f}  ({n_detected}/{n_action})')
    print(f'CLASSIFICATION ACC (detected)     = {cls_acc_detected:.4f}  ({n_correct}/{n_detected})')
    print(f'CLASSIFICATION ACC (all actions)  = {cls_acc_all:.4f}  ({n_correct}/{n_action})')
    print(f'FALSE ALARM RATE (background)     = {fa_rate:.4f}  ({n_false_alarm}/{n_bg})')
    print()

    # ---- per-class breakdown ----
    print(f'{"class":<16}{"det_rate":>10}{"cls_acc":>10}{"support":>9}')
    for i in action:
        sup = cm[i, :].sum()
        det = sup - cm[i, bg]
        d = det / sup if sup else 0.0
        a = cm[i, i] / det if det else 0.0
        print(f'{labels[i]:<16}{d:>10.3f}{a:>10.3f}{sup:>9}')


if __name__ == '__main__':
    main()
