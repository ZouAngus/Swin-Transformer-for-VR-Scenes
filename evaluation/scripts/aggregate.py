#!/usr/bin/env python
"""Aggregate eval_long_video.py results across all long test videos.

Repository copy of work_dirs/long_eval_18cls/aggregate.py, with two fixes:

  1. The evaluation directory is an argument instead of the hard-coded
     'eval/'. The original could only aggregate the uncalibrated baseline
     run, which is why the summary checked into work_dirs did not match the
     9.3% duration error quoted in the report (that number comes from the
     background-calibrated run, eval_supp0.1/).
  2. --exclude-label lets you reproduce the report's activity vocabulary,
     where Walking is background by design and therefore not an activity.

Usage
-----
    python aggregate.py eval_supp0.1 --exclude-label Walking
    python aggregate.py eval                      # uncalibrated baseline
    python aggregate.py eval_supp0.1 --raw-out raw.json

Run it from work_dirs/long_eval_18cls (or pass an absolute path).
"""
import argparse
import glob
import json
import statistics
from collections import defaultdict

FPS = 30.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('eval_dir', nargs='?', default='eval',
                    help="directory holding <video>/eval_results.json (default: eval)")
    ap.add_argument('--exclude-label', action='append', default=[],
                    help='drop segments with this ground-truth label; repeatable')
    ap.add_argument('--raw-out', default=None,
                    help='optional path to dump the raw per-segment records as JSON')
    args = ap.parse_args()

    files = sorted(glob.glob(f'{args.eval_dir}/*/eval_results.json'))
    if not files:
        raise SystemExit(f'no eval_results.json found under {args.eval_dir}/')

    segs, mious = [], []
    for fp in files:
        r = json.load(open(fp))
        mious.append((r['video_name'], r['mean_iou']))
        for d in r['duration_error_details']:
            d['video'] = r['video_name']
            d['scene'] = r['scene']
            segs.append(d)

    total_before = len(segs)
    if args.exclude_label:
        segs = [s for s in segs if s['label'] not in args.exclude_label]

    n = len(segs)
    errs = [s['rel_error'] for s in segs]

    print(f'eval dir:          {args.eval_dir}')
    print(f'videos evaluated:  {len(files)}')
    if args.exclude_label:
        print(f'GT segments total: {n}  (excluded {total_before - n}: '
              f'{", ".join(args.exclude_label)})')
    else:
        print(f'GT segments total: {n}')
    print(f'mean rel error:    {statistics.mean(errs) * 100:.1f}%')
    print(f'median rel error:  {statistics.median(errs) * 100:.1f}%')
    for thr in (0.1, 0.2, 0.3):
        k = sum(1 for s in segs if s['rel_error'] < thr)
        print(f'rel error < {int(thr * 100)}%: {k / n * 100:.1f}% ({k}/{n})')

    for lo, hi, tag in ((0, 90, '<3s'), (90, 10 ** 9, '>=3s')):
        sub = [s for s in segs if lo <= s['gt_duration'] < hi]
        if sub:
            k = sum(1 for s in sub if s['rel_error'] < 0.1)
            med = statistics.median([s['rel_error'] for s in sub])
            print(f'segments {tag}: n={len(sub)}  <10%: {k / len(sub) * 100:.1f}%  '
                  f'median err: {med * 100:.1f}%')

    print()
    print(f'{"class":<16}{"n":>5}{"<10%":>8}{"median":>9}')
    bycls = defaultdict(list)
    for s in segs:
        bycls[s['label']].append(s)
    for lbl, ss in sorted(bycls.items(), key=lambda kv: -len(kv[1])):
        k = sum(1 for s in ss if s['rel_error'] < 0.1)
        med = statistics.median([s['rel_error'] for s in ss])
        print(f'{lbl:<16}{len(ss):>5}{k / len(ss) * 100:>7.1f}%{med * 100:>8.1f}%')

    print()
    print(f'mean mIoU over videos (bg incl, thr 0): '
          f'{statistics.mean(m for _, m in mious):.4f}')

    if args.raw_out:
        json.dump({'segments': segs, 'mious': mious}, open(args.raw_out, 'w'))
        print(f'\nraw records written to {args.raw_out}')


if __name__ == '__main__':
    main()
