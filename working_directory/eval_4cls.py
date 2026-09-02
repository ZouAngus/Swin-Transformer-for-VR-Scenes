#!/usr/bin/env python
"""Per-class P/R + confusion matrix from a tools/test.py preds.pkl dump."""
import argparse
import pickle
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--preds', required=True, help='pkl from tools/test.py --dump')
    p.add_argument('--ann', required=True, help='val_list.txt (path label)')
    p.add_argument('--labels', required=True, help='label_4cls.txt')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.preds, 'rb') as f:
        preds = pickle.load(f)
    with open(args.ann) as f:
        gts = [int(line.strip().rsplit(' ', 1)[1]) for line in f if line.strip()]
    with open(args.labels) as f:
        labels = [l.strip() for l in f if l.strip()]

    assert len(preds) == len(gts), f'{len(preds)} preds vs {len(gts)} gts'
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    pred_labels = []
    for p, gt in zip(preds, gts):
        score = p['pred_score'] if isinstance(p, dict) else p.pred_score
        if hasattr(score, 'cpu'):
            score = score.cpu().numpy()
        score = np.asarray(score)
        pred = int(score.argmax())
        pred_labels.append(pred)
        cm[gt, pred] += 1

    total = cm.sum()
    acc = np.trace(cm) / total
    print(f'samples={total}  top-1 acc={acc:.4f}')
    print()
    header = 'gt\\pred  ' + '  '.join(f'{l[:8]:>8}' for l in labels) + '   support'
    print(header)
    for i, l in enumerate(labels):
        row = '  '.join(f'{cm[i,j]:>8}' for j in range(n))
        print(f'{l[:8]:<8}  {row}   {cm[i].sum():>7}')

    print()
    print(f'{"class":<14}{"prec":>8}{"rec":>8}{"f1":>8}{"sup":>6}')
    for i, l in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f'{l:<14}{prec:>8.3f}{rec:>8.3f}{f1:>8.3f}{cm[i].sum():>6}')


if __name__ == '__main__':
    main()
