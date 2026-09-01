#!/usr/bin/env python
"""Aggregate eval_long_video.py results across all test long videos."""
import json, glob, statistics
from collections import defaultdict

FPS = 30.0
files = sorted(glob.glob('eval/*/eval_results.json'))
segs = []
mious = []
for fp in files:
    r = json.load(open(fp))
    mious.append((r['video_name'], r['mean_iou']))
    for d in r['duration_error_details']:
        d['video'] = r['video_name']
        d['scene'] = r['scene']
        segs.append(d)

n = len(segs)
def ratio(pred):
    k = sum(1 for s in segs if pred(s))
    return k, n, k/n if n else 0

print(f'videos evaluated: {len(files)}')
print(f'GT segments total: {n}')
errs = [s['rel_error'] for s in segs]
print(f'mean rel error:   {statistics.mean(errs)*100:.1f}%')
print(f'median rel error: {statistics.median(errs)*100:.1f}%')
for thr in (0.1, 0.2, 0.3):
    k,_,rt = ratio(lambda s, t=thr: s['rel_error'] < t)
    print(f'rel error < {int(thr*100)}%: {rt*100:.1f}% ({k}/{n})')
# duration buckets
for lo, hi, tag in ((0, 90, '<3s'), (90, 10**9, '>=3s')):
    sub = [s for s in segs if lo <= s['gt_duration'] < hi]
    if sub:
        k = sum(1 for s in sub if s['rel_error'] < 0.1)
        med = statistics.median([s['rel_error'] for s in sub])
        print(f'segments {tag}: n={len(sub)}  <10%: {k/len(sub)*100:.1f}%  median err: {med*100:.1f}%')
# per class
print()
print(f'{"class":<16}{"n":>5}{"<10%":>8}{"median":>9}')
bycls = defaultdict(list)
for s in segs: bycls[s['label']].append(s)
for lbl, ss in sorted(bycls.items(), key=lambda kv: -len(kv[1])):
    k = sum(1 for s in ss if s['rel_error'] < 0.1)
    med = statistics.median([s['rel_error'] for s in ss])
    print(f'{lbl:<16}{len(ss):>5}{k/len(ss)*100:>7.1f}%{med*100:>8.1f}%')
print()
print(f'mean mIoU over videos (bg incl, thr 0): {statistics.mean(m for _, m in mious):.4f}')
json.dump({'segments': segs, 'mious': mious}, open('aggregate_raw.json', 'w'))
