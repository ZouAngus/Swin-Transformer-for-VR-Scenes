# Copyright (c) OpenMMLab. All rights reserved.
"""Long-video demo for the 4-class fine-tuned recognizer (v2).

Drop-in superset of demo/long_video_demo.py:
  - keeps original CLI: --config --checkpoint --video_path --label
    --out_file --input-step --stride --threshold --device
  - keeps original outputs:
        DemoData_<TIMESTAMP>.json  (per-frame top-K scores)
        ActionData_<TIMESTAMP>.json (via format_json_to_action.py)
        annotated mp4 (when --out_file ends with .mp4)
  - temporal post-processing for demo robustness:
        --smooth-k        moving average over the last K window softmaxes
        --hysteresis      require N consecutive smoothed picks to switch state
        --stand-index     fallback class when below threshold
        --decision-overlay
                          additionally render the smoothed/gated decision
                          (label + prob) as a bottom overlay on the video
        --decision-json   optional path; dumps a compact summary
                          {windows: [...], segments: [...]} that is easier
                          to consume than DemoData
  - NEW v2: --min-segment-duration
        After per-window decisions are produced, consolidate them into
        contiguous segments and drop any non-Stand segment whose duration
        in seconds is below the threshold (i.e. relabel its windows back
        to --stand-index). Then RE-RENDER the video overlay and write all
        downstream JSON outputs (DemoData / ActionData / decisions.json)
        using the filtered labels. This guarantees that the rendered MP4,
        the per-frame DemoData, and the segment-level ActionData all stay
        consistent with the filtered decisions.

Implementation note (v2 only):
  Pass A runs inference and collects per-window decisions (label probs).
  After all windows are processed, we apply the segment-duration filter
  to mutate decision labels for short non-Stand segments. Then Pass B
  re-opens the source video and writes the annotated mp4 using the
  filtered label for each frame. JSON sidecars are produced from the
  filtered decision list.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import deque
from operator import itemgetter

import cv2
import mmengine
import numpy as np
import torch
from mmengine import Config, DictAction
from mmengine.dataset import Compose

from mmaction.apis import inference_recognizer, init_recognizer

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
EXPORT_FPS = 30.0
FONTSCALE = 1
THICKNESS = 1
LINETYPE = 1

EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]

TIMESTAMP_PATTERN = re.compile(
    r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})-(?P<hour>\d{2})-(?P<minute>\d{2})-(?P<second>\d{2})$'
)
FORMAT_JSON_TO_ACTION_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'format_json_to_action.py')
DEFAULT_OUT_DIR = '/home/zhuyusi/my_mmaction2/demo/digitaltwins/results'


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 long-video demo for the 4-class recognizer '
                    '(v2 with short-segment filtering).')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file/url')
    parser.add_argument(
        '-inputfile',
        '--inputfile',
        '--video_path',
        dest='video_path',
        help='video file/url')
    parser.add_argument('--label', help='label file')
    parser.add_argument('--out_file', help='output result file in video/json')
    parser.add_argument(
        '--input-step', type=int, default=1,
        help='input step for sampling frames')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--threshold', type=float, default=0.01,
        help='top-K score threshold used for legacy DemoData entries; for '
             'decision gating prefer --decision-threshold')
    parser.add_argument(
        '--stride', type=float, default=0,
        help='prediction stride = stride * sample_length; 0 means stride 1')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction, default={},
        help='override some settings in the used config')
    parser.add_argument(
        '--label-color', nargs='+', type=int, default=(255, 255, 255),
        help='font color (B, G, R) of the labels in output video')
    parser.add_argument(
        '--msg-color', nargs='+', type=int, default=(128, 128, 128),
        help='font color (B, G, R) of the messages in output video')

    # ---- temporal post-processing extensions ----
    parser.add_argument(
        '--smooth-k', type=int, default=1,
        help='moving-average window over softmax probs (1 = no smoothing)')
    parser.add_argument(
        '--decision-threshold', type=float, default=0.0,
        help='min smoothed prob for non-stand classes')
    parser.add_argument(
        '--hysteresis', type=int, default=1,
        help='consecutive identical smoothed picks required to switch state')
    parser.add_argument(
        '--stand-index', type=int, default=0,
        help='label index treated as the default/fallback state')
    parser.add_argument(
        '--decision-overlay', action='store_true',
        help='also render the smoothed decision as a bottom overlay')
    parser.add_argument(
        '--decision-json', default=None,
        help='optional path for a compact decisions+segments JSON dump')

    # NEW in v2:
    parser.add_argument(
        '--min-segment-duration', type=float, default=0.1,
        help='drop non-Stand segments whose duration in seconds is below '
             'this threshold (relabel their windows to --stand-index). '
             '0.0 disables. Default 0.1 removes 0s single-window FP '
             'transients. Set to 0 to keep the raw decision stream.')
    parser.add_argument(
        '--sample-length', type=int, default=32,
        help='temporal window length passed to the recognizer; default 32 '
             'matches the recipe used during training')

    args = parser.parse_args()
    return args


def get_timestamp_token(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    match = TIMESTAMP_PATTERN.search(video_name)
    if match is None:
        raise ValueError(
            'video_path must end with a timestamp like YYYY-MM-DD-HH-MM-SS')
    parts = match.groupdict()
    return (
        f"{parts['year']}{parts['month']}{parts['day']}_"
        f"{parts['hour']}{parts['minute']}{parts['second']}"
    )


def resolve_json_output_paths(video_path, out_file):
    timestamp_token = get_timestamp_token(video_path)
    output_dir = os.path.dirname(out_file) if out_file else ''
    if not output_dir:
        output_dir = os.path.dirname(video_path)
    demo_json = os.path.join(output_dir,
                             f'DemoData_{timestamp_token}.json')
    action_json = os.path.join(output_dir,
                               f'ActionData_{timestamp_token}.json')
    return demo_json, action_json


def build_action_formatter_command(script_path, demo_json_path,
                                   action_json_path, fps):
    if not os.path.isfile(script_path):
        raise FileNotFoundError(
            f'Action formatter script not found: {script_path}')
    with open(script_path, 'r', encoding='utf-8') as script_file:
        script_source = script_file.read()
    if '--input-json' in script_source and '--output-json' in script_source:
        return [sys.executable, script_path, '--input-json', demo_json_path,
                '--output-json', action_json_path, "--fps", str(fps)]
    if '--input' in script_source and '--output' in script_source:
        return [sys.executable, script_path, '--input', demo_json_path,
                '--output', action_json_path, "--fps", str(fps)]
    return [sys.executable, script_path, demo_json_path, action_json_path, "--fps", str(fps)]


def run_action_formatter(demo_json_path, action_json_path, fps):
    formatter_cmd = build_action_formatter_command(
        FORMAT_JSON_TO_ACTION_SCRIPT, demo_json_path, action_json_path, fps)
    subprocess.run(formatter_cmd, check=True)


def render_topk_overlay(frame, results, threshold, label_color, text_info):
    text_info.clear()
    for i, result in enumerate(results):
        selected_label, score = result
        if score < threshold:
            break
        location = (0, 40 + i * 20)
        text = f'{selected_label}: {round(float(score), 2)}'
        text_info[location] = text
        cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                    label_color, THICKNESS, LINETYPE)
    return text_info


def reapply_topk_overlay(frame, text_info, label_color):
    for location, text in text_info.items():
        cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                    label_color, THICKNESS, LINETYPE)


def render_decision_overlay(frame, decision_label, decision_prob, label_color):
    h, w = frame.shape[:2]
    bar_h = 36
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    text = f'{decision_label}  p={decision_prob:.2f}'
    cv2.putText(frame, text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def consolidate_segments(decisions, fps, sample_length):
    """Consolidate per-window decisions into segments."""
    segments = []
    cur = None
    for d in decisions:
        center_s = d['frame_ind'] / fps - (sample_length / fps) / 2.0
        center_s = max(0.0, round(center_s, 3))
        lab = d['label']
        if cur is None or lab != cur['label']:
            if cur is not None:
                segments.append(cur)
            cur = dict(label=lab, start_s=center_s, end_s=center_s,
                       probs=[d['prob']], window_indices=[len(segments)])
            # also track which decisions belong to this segment
            cur['decision_indices'] = []
        else:
            cur['end_s'] = center_s
            cur['probs'].append(d['prob'])
    # the above is buggy w.r.t. decision_indices; rebuild cleanly:
    segments = []
    cur = None
    for di, d in enumerate(decisions):
        center_s = d['frame_ind'] / fps - (sample_length / fps) / 2.0
        center_s = max(0.0, round(center_s, 3))
        lab = d['label']
        if cur is None or lab != cur['label']:
            if cur is not None:
                segments.append(cur)
            cur = dict(label=lab, start_s=center_s, end_s=center_s,
                       probs=[d['prob']], decision_indices=[di])
        else:
            cur['end_s'] = center_s
            cur['probs'].append(d['prob'])
            cur['decision_indices'].append(di)
    if cur is not None:
        segments.append(cur)
    for s in segments:
        s['avg_prob'] = round(float(np.mean(s['probs'])), 4)
    return segments


def apply_min_duration_filter(decisions, fps, sample_length,
                              min_duration_s, stand_label):
    """Mutate decisions[]: any non-Stand segment with duration < min_duration_s
    gets its windows relabeled to stand_label (and prob set to that class's
    smoothed prob if available, else kept). Returns count of dropped segments.
    """
    if min_duration_s <= 0:
        return 0
    segments = consolidate_segments(decisions, fps, sample_length)
    dropped = 0
    for seg in segments:
        if seg['label'] == stand_label:
            continue
        dur = seg['end_s'] - seg['start_s']
        if dur < min_duration_s:
            for di in seg['decision_indices']:
                d = decisions[di]
                # relabel to Stand and adjust prob to the Stand class prob
                d['label'] = stand_label
                stand_prob = d.get('smoothed', {}).get(stand_label, None)
                if stand_prob is not None:
                    d['prob'] = round(float(stand_prob), 4)
            dropped += 1
    return dropped


def apply_burst_suppression(decisions, fps, sample_length, stand_label,
                            conf_thresh=0.9, min_cluster_size=3,
                            max_span_s=15.0):
    """Suppress clusters of low-confidence non-Stand segments."""
    segments = consolidate_segments(decisions, fps, sample_length)
    non_stand_segs = [s for s in segments if s['label'] != stand_label]

    if len(non_stand_segs) < min_cluster_size:
        return 0

    suppressed_segs = []
    i = 0
    while i < len(non_stand_segs):
        if non_stand_segs[i]['avg_prob'] >= conf_thresh:
            i += 1
            continue
        cluster = [i]
        j = i + 1
        while j < len(non_stand_segs):
            if non_stand_segs[j]['avg_prob'] >= conf_thresh:
                break
            span = non_stand_segs[j]['end_s'] - non_stand_segs[cluster[0]]['start_s']
            if span > max_span_s:
                break
            cluster.append(j)
            j += 1

        if len(cluster) >= min_cluster_size:
            suppressed_segs.extend(cluster)
            i = j
        else:
            i += 1

    if not suppressed_segs:
        return 0

    suppressed_count = 0
    for si in suppressed_segs:
        seg = non_stand_segs[si]
        for di in seg['decision_indices']:
            d = decisions[di]
            d['label'] = stand_label
            stand_prob = d.get('smoothed', {}).get(stand_label, None)
            if stand_prob is not None:
                d['prob'] = round(float(stand_prob), 4)
        suppressed_count += 1

    return suppressed_count


def show_results(model, data, label, args):
    """Pass A: run inference, collect per-window decisions (no video write)."""
    frame_queue = deque(maxlen=args.sample_length)

    prob_hist = deque(maxlen=max(1, args.smooth_k))
    pending_label = args.stand_index
    pending_count = 0
    current_label = args.stand_index
    decisions = []

    cap = cv2.VideoCapture(args.video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = EXPORT_FPS

    prog_bar = mmengine.ProgressBar(num_frames)
    backup_frames = []

    ind = 0
    while ind < num_frames:
        ind += 1
        prog_bar.update()
        ret, frame = cap.read()
        if frame is None:
            continue
        backup_frames.append(np.array(frame)[:, :, ::-1])
        if ind == args.sample_length:
            frame_queue.extend(backup_frames)
            backup_frames = []
        elif ((len(backup_frames) == args.input_step
               and ind > args.sample_length) or ind == num_frames):
            chosen_frame = backup_frames[0]
            backup_frames = []
            frame_queue.append(chosen_frame)

        ret_pred, scores = inference(model, data, args, frame_queue)

        if ret_pred:
            scores_np = np.asarray(scores, dtype=np.float64)
            prob_hist.append(scores_np)
            smoothed = np.mean(np.stack(prob_hist), axis=0)
            top_idx = int(np.argmax(smoothed))
            top_prob = float(smoothed[top_idx])

            if (args.decision_threshold > 0
                    and top_idx != args.stand_index
                    and top_prob < args.decision_threshold):
                gated = args.stand_index
            else:
                gated = top_idx

            if args.hysteresis <= 1:
                current_label = gated
            else:
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

            decisions.append(dict(
                frame_ind=int(ind),
                label=label[current_label],
                prob=round(current_prob, 4),
                smoothed={label[i]: round(float(smoothed[i]), 4)
                          for i in range(len(label))},
            ))

    cap.release()
    return decisions, fps, num_frames


def render_pass(decisions, fps, num_frames, label, args):
    """Pass B: re-open video, render annotated mp4 using (possibly filtered)
    decision labels, and write JSON sidecars."""
    is_video_out = not args.out_file.endswith('.json')

    cap = cv2.VideoCapture(args.video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = (frame_width, frame_height)
    video_writer = (cv2.VideoWriter(args.out_file, fourcc, fps, frame_size)
                    if is_video_out else None)

    # decisions are emitted starting at frame == sample_length, then every
    # input_step frames. We iterate decisions in order and assign each to a
    # contiguous range of output frames.
    msg = 'Preparing action recognition ...'
    text_info = {}
    out_json = {}

    # Build a per-frame label/prob mapping using "latest decision wins" semantics
    # (matches v1 which used current_label / current_prob each frame).
    frame_to_dec = [None] * (num_frames + 2)
    last_dec = None
    di = 0
    for f_idx in range(1, num_frames + 1):
        while di < len(decisions) and decisions[di]['frame_ind'] <= f_idx:
            last_dec = decisions[di]
            di += 1
        frame_to_dec[f_idx] = last_dec

    prog_bar = mmengine.ProgressBar(num_frames)
    ind = 0
    while ind < num_frames:
        ind += 1
        prog_bar.update()
        ret, frame = cap.read()
        if frame is None:
            continue

        dec = frame_to_dec[ind]
        if is_video_out:
            if dec is not None:
                # synthesize a results list for top-K overlay from smoothed probs
                smoothed_list = [dec['smoothed'][l] for l in label]
                scores_tuples = tuple(zip(label, smoothed_list))
                scores_sorted = sorted(scores_tuples, key=itemgetter(1),
                                       reverse=True)
                results = scores_sorted[:min(len(label), 5)]
                text_info = render_topk_overlay(
                    frame, results, args.threshold, args.label_color,
                    text_info)
                if args.decision_overlay:
                    render_decision_overlay(
                        frame, dec['label'], dec['prob'], args.label_color)
            elif text_info:
                reapply_topk_overlay(frame, text_info, args.label_color)
                if args.decision_overlay:
                    render_decision_overlay(
                        frame, label[args.stand_index], 0.0,
                        args.label_color)
            else:
                cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE,
                            args.msg_color, THICKNESS, LINETYPE)
            video_writer.write(frame)
        else:
            # json out mode: only write DemoData json
            if dec is not None:
                smoothed_list = [dec['smoothed'][l] for l in label]
                scores_tuples = tuple(zip(label, smoothed_list))
                scores_sorted = sorted(scores_tuples, key=itemgetter(1),
                                       reverse=True)
                results = scores_sorted[:min(len(label), 5)]
                entry = {}
                for i, (sel_label, score) in enumerate(results):
                    if score < args.threshold:
                        break
                    entry[i + 1] = f'{sel_label}: {round(float(score), 2)}'
                entry['_smoothed'] = dec['smoothed']
                entry['_decision'] = dec['label']
                entry['_decision_prob'] = dec['prob']
                out_json[ind] = entry
                text_info = {1: entry.get(1, '')}
            elif text_info:
                out_json[ind] = text_info
            else:
                out_json[ind] = msg

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    if args.out_file.endswith('.json'):
        with open(args.out_file, 'w') as js:
            json.dump(out_json, js)
        run_action_formatter(args.out_file, args.action_out_file, fps)

    # Always also dump DemoData/ActionData siblings when video output mode is used
    if not args.out_file.endswith('.json') and getattr(
            args, 'demo_json_out', None):
        legacy_json = {}
        for d in decisions:
            ind_key = d['frame_ind']
            entry = {1: f"{d['label']}: {round(d['prob'], 2)}",
                     '_smoothed': d['smoothed'],
                     '_decision': d['label'],
                     '_decision_prob': d['prob']}
            legacy_json[ind_key] = entry
        with open(args.demo_json_out, 'w') as js:
            json.dump(legacy_json, js)
        run_action_formatter(args.demo_json_out, args.action_json_out, fps)

    if args.decision_json:
        seg = consolidate_segments(decisions, fps, args.sample_length)
        for s in seg:
            if 'probs' in s:
                del s['probs']
            if 'decision_indices' in s:
                del s['decision_indices']
        os.makedirs(os.path.dirname(os.path.abspath(args.decision_json)) or '.',
                    exist_ok=True)
        with open(args.decision_json, 'w') as js:
            json.dump(dict(
                video=args.video_path,
                fps=fps,
                num_frames=num_frames,
                sample_length=args.sample_length,
                input_step=args.input_step,
                stride=args.stride,
                smooth_k=args.smooth_k,
                decision_threshold=args.decision_threshold,
                hysteresis=args.hysteresis,
                stand_index=args.stand_index,
                min_segment_duration=args.min_segment_duration,
                labels=label,
                windows=decisions,
                segments=seg,
            ), js, indent=2)


def inference(model, data, args, frame_queue):
    if len(frame_queue) != args.sample_length:
        return False, None
    cur_windows = list(np.array(frame_queue))
    if data['img_shape'] is None:
        data['img_shape'] = frame_queue[0].shape[:2]
    resized_windows = [cv2.resize(frame, (224, 224)) for frame in cur_windows]
    cur_data = data.copy()
    cur_data.update(dict(
        array=resized_windows,
        modality='RGB',
        frame_inds=np.arange(args.sample_length)))
    result = inference_recognizer(
        model, cur_data, test_pipeline=args.test_pipeline)
    scores = result.pred_score.tolist()
    if args.stride > 0:
        pred_stride = int(args.sample_length * args.stride)
        for _ in range(pred_stride):
            frame_queue.popleft()
    return True, scores


def main():
    args = parse_args()
    args.device = torch.device(args.device)

    # Default output paths land under demo/digitaltwins/results/ when the
    # caller did not specify them explicitly. Video stem mirrors the input.
    video_stem = os.path.splitext(os.path.basename(args.video_path))[0]
    if not args.out_file:
        os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
        args.out_file = os.path.join(DEFAULT_OUT_DIR, f'{video_stem}.mp4')
    else:
        out_dir = os.path.dirname(args.out_file)
        if not out_dir:
            os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
            args.out_file = os.path.join(DEFAULT_OUT_DIR, args.out_file)
        else:
            os.makedirs(out_dir, exist_ok=True)
    if not args.decision_json:
        args.decision_json = os.path.join(
            os.path.dirname(args.out_file),
            f'{video_stem}_decisions.json')

    # always-on JSON sidecar paths (mirror v1 behavior)
    is_video_out = not (args.out_file or '').endswith('.json')
    if is_video_out:
        args.demo_json_out, args.action_json_out = resolve_json_output_paths(
            args.video_path, args.out_file)
    else:
        args.demo_json_out = None
        args.action_json_out = None
        if args.out_file.endswith('.json'):
            _, args.action_out_file = resolve_json_output_paths(
                args.video_path, args.out_file)

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    model = init_recognizer(cfg, args.checkpoint, device=args.device)
    data = dict(img_shape=None, modality='RGB', label=-1)
    with open(args.label, 'r') as f:
        label = [line.strip() for line in f if line.strip()]

    cfg = model.cfg
    sample_length = 0
    pipeline = cfg.test_pipeline
    pipeline_ = pipeline.copy()
    for step in pipeline:
        if 'SampleFrames' in step['type']:
            sample_length = step['clip_len'] * step['num_clips']
            data['num_clips'] = step['num_clips']
            data['clip_len'] = step['clip_len']
            pipeline_.remove(step)
        if step['type'] in EXCLUED_STEPS:
            pipeline_.remove(step)
    pipeline_.insert(1, dict(type='ArrayDecode'))
    test_pipeline = Compose(pipeline_)

    assert sample_length > 0
    args.sample_length = sample_length
    args.test_pipeline = test_pipeline

    # Pass A: run inference
    print('[Pass A] running inference...', flush=True)
    decisions, fps, num_frames = show_results(model, data, label, args)

    # Apply min-segment-duration filter (mutates decisions[].label in place)
    stand_label = label[args.stand_index]
    dropped = apply_min_duration_filter(
        decisions, fps, args.sample_length,
        args.min_segment_duration, stand_label)
    print(f'\n[Filter] min-segment-duration={args.min_segment_duration}s, '
          f'dropped {dropped} short non-Stand segments', flush=True)


    # Apply burst suppression (suppress clusters of low-confidence FP actions)
    burst_suppressed = apply_burst_suppression(
        decisions, fps, args.sample_length, stand_label,
        conf_thresh=0.9, min_cluster_size=3, max_span_s=15.0)
    if burst_suppressed:
        print(f'[Filter] burst-suppression: suppressed {burst_suppressed} '
              f'segments in low-confidence clusters', flush=True)
    # Pass B: render annotated video + write JSON sidecars
    print('[Pass B] rendering annotated video and writing JSON...', flush=True)
    render_pass(decisions, fps, num_frames, label, args)


if __name__ == '__main__':
    main()
