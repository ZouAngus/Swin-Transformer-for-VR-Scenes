#!/usr/bin/env python
"""Plots for detection/classification evaluation, matching the style of the
2025-08-06 handover report (per-class accuracy bars + log-scale confusion matrix).

Usage:
  python tools/plot_det_cls.py \
      --preds work_dirs/base18_eval/preds.pkl \
      --ann data/myvideo/myvideo_val_list.txt \
      --labels data/myvideo/classInd.txt \
      --bg-index 0 \
      --out-dir work_dirs/base18_eval

Outputs (in --out-dir):
  recognition_accuracy.png   per-class recognition accuracy bars (as in the old report)
  confusion_matrix.png       confusion matrix, log-scale coloring (as in the old report)
  det_cls_rates.png          per-class detection rate & classification accuracy
                             with the 90% / 85% requirement lines
"""
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--preds', required=True)
    p.add_argument('--ann', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--bg-index', type=int, default=0)
    p.add_argument('--out-dir', required=True)
    return p.parse_args()


def load_labels(path):
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            labels.append(parts[1] if len(parts) == 2 and parts[0].isdigit() else line)
    return labels


def build_cm(preds_path, ann_path, n):
    with open(preds_path, 'rb') as f:
        preds = pickle.load(f)
    with open(ann_path) as f:
        gts = [int(l.strip().rsplit(' ', 1)[1]) for l in f if l.strip()]
    assert len(preds) == len(gts), f'{len(preds)} preds vs {len(gts)} gts'
    cm = np.zeros((n, n), dtype=int)
    for p, gt in zip(preds, gts):
        score = p['pred_score'] if isinstance(p, dict) else p.pred_score
        if hasattr(score, 'cpu'):
            score = score.cpu().numpy()
        cm[gt, int(np.asarray(score).argmax())] += 1
    return cm


def plot_recognition_accuracy(cm, labels, bg, out):
    names = ['Background' if i == bg else labels[i] for i in range(len(labels))]
    accs = [cm[i, i] / cm[i].sum() if cm[i].sum() else 0 for i in range(len(labels))]
    names.append('Accuracy')
    accs.append(np.trace(cm) / cm.sum())
    sups = [cm[i].sum() for i in range(len(labels))] + [cm.sum()]
    hits = [cm[i, i] for i in range(len(labels))] + [np.trace(cm)]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(names) + 1))
    y = np.arange(len(names))[::-1]
    ax.barh(y, accs, color='black', height=0.62)
    ax.set_xlim(0, 1.38)
    for yi, a, h, s in zip(y, accs, hits, sups):
        ax.text(1.02, yi, f'{a * 100:6.2f}% ({h}/{s})', va='center',
                fontsize=9, family='monospace')
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9, family='monospace')
    ax.axvline(1.0, color='gray', lw=0.8)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title('Recognition Accuracy (18-class base model, test persons 11-15)')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm, labels, out):
    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(13, 9))
        sns.heatmap(np.log1p(cm), annot=cm, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'log(1+count)'}, ax=ax,
                    annot_kws={'fontsize': 8})
    except ImportError:
        fig, ax = plt.subplots(figsize=(13, 9))
        im = ax.imshow(np.log1p(cm), cmap='Blues')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=7,
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        fig.colorbar(im, label='log(1+count)')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix (Log-scale coloring)')
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_det_cls(cm, labels, bg, out):
    action = [i for i in range(len(labels)) if i != bg]
    det, cls_acc, names = [], [], []
    for i in action:
        sup = cm[i].sum()
        d = sup - cm[i, bg]
        det.append(d / sup if sup else 0)
        cls_acc.append(cm[i, i] / d if d else 0)
        names.append(labels[i])
    # overall
    n_action = cm[action, :].sum()
    n_det = n_action - cm[action, bg].sum()
    n_cor = sum(cm[i, i] for i in action)
    names.append('OVERALL')
    det.append(n_det / n_action)
    cls_acc.append(n_cor / n_det)

    y = np.arange(len(names))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(names) + 1.2), sharey=True)
    for ax, vals, req, title in (
            (axes[0], det, 0.90, 'Detection rate (req >= 90%)'),
            (axes[1], cls_acc, 0.85, 'Classification acc. among detected (req >= 85%)')):
        colors = ['#b22222' if v < req else '#2f6f2f' for v in vals]
        colors[-1] = 'black'
        ax.barh(y, vals, color=colors, height=0.62)
        ax.axvline(req, color='red', ls='--', lw=1)
        ax.set_xlim(0, 1.02)
        for yi, v in zip(y, vals):
            ax.text(min(v + 0.01, 0.86), yi, f'{v * 100:.1f}%', va='center', fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=9, family='monospace')
    fig.suptitle('18-class base model - downstream requirement check (clip level)', y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    labels = load_labels(args.labels)
    cm = build_cm(args.preds, args.ann, len(labels))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_recognition_accuracy(cm, labels, args.bg_index, out / 'recognition_accuracy.png')
    plot_confusion_matrix(cm, labels, out / 'confusion_matrix.png')
    plot_det_cls(cm, labels, args.bg_index, out / 'det_cls_rates.png')
    print(f'saved 3 figures to {out}')


if __name__ == '__main__':
    main()
