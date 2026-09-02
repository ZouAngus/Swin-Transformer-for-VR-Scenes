# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import json
import os
import random
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 predict different labels in a long video demo')
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
        '--input-step',
        type=int,
        default=1,
        help='input step for sampling frames')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help='recognition score threshold')
    parser.add_argument(
        '--bg-scale',
        type=float,
        default=1.0,
        help='scale factor applied to the background (Stand) probability '
             'before argmax; 1.0 = disabled. Validated for the 18-class '
             'base model only (see EVAL_detection_classification.md); '
             'do NOT use with the 4-class digital-twin models.')
    parser.add_argument(
        '--bg-index',
        type=int,
        default=0,
        help='index of the background class (default 0 = Stand)')
    parser.add_argument(
        '--stride',
        type=float,
        default=0,
        help=('the prediction stride equals to stride * sample_length '
              '(sample_length indicates the size of temporal window from '
              'which you sample frames, which equals to '
              'clip_len x frame_interval), if set as 0, the '
              'prediction stride is 1'))
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--label-color',
        nargs='+',
        type=int,
        default=(255, 255, 255),
        help='font color (B, G, R) of the labels in output video')
    parser.add_argument(
        '--msg-color',
        nargs='+',
        type=int,
        default=(128, 128, 128),
        help='font color (B, G, R) of the messages in output video')
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
                                   action_json_path):
    if not os.path.isfile(script_path):
        raise FileNotFoundError(
            f'Action formatter script not found: {script_path}')

    with open(script_path, 'r', encoding='utf-8') as script_file:
        script_source = script_file.read()

    if '--input-json' in script_source and '--output-json' in script_source:
        return [
            sys.executable, script_path, '--input-json', demo_json_path,
            '--output-json', action_json_path
        ]

    if '--input' in script_source and '--output' in script_source:
        return [
            sys.executable, script_path, '--input', demo_json_path, '--output',
            action_json_path
        ]

    return [sys.executable, script_path, demo_json_path, action_json_path]


def run_action_formatter(demo_json_path, action_json_path):
    formatter_cmd = build_action_formatter_command(
        FORMAT_JSON_TO_ACTION_SCRIPT, demo_json_path, action_json_path)
    subprocess.run(formatter_cmd, check=True)


def show_results_video(result_queue,
                       text_info,
                       thr,
                       msg,
                       frame,
                       video_writer,
                       label_color=(255, 255, 255),
                       msg_color=(128, 128, 128)):
    if len(result_queue) != 0:
        text_info = {}
        results = result_queue.popleft()
        for i, result in enumerate(results):
            selected_label, score = result
            if score < thr:
                break
            location = (0, 40 + i * 20)
            text = selected_label + ': ' + str(round(score, 2))
            text_info[location] = text
            cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                        label_color, THICKNESS, LINETYPE)
    elif len(text_info):
        for location, text in text_info.items():
            cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                        label_color, THICKNESS, LINETYPE)
    else:
        cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, msg_color,
                    THICKNESS, LINETYPE)
    video_writer.write(frame)
    return text_info


def get_results_json(result_queue, text_info, thr, msg, ind, out_json):
    if len(result_queue) != 0:
        text_info = {}
        results = result_queue.popleft()
        for i, result in enumerate(results):
            selected_label, score = result
            if score < thr:
                break
            text_info[i + 1] = selected_label + ': ' + str(round(score, 2))
        out_json[ind] = text_info
    elif len(text_info):
        out_json[ind] = text_info
    else:
        out_json[ind] = msg
    return text_info, out_json


def show_results(model, data, label, args):
    frame_queue = deque(maxlen=args.sample_length)
    result_queue = deque(maxlen=1)

    cap = cv2.VideoCapture(args.video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    msg = 'Preparing action recognition ...'
    text_info = {}
    out_json = {}
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = (frame_width, frame_height)

    ind = 0
    video_writer = None if args.out_file.endswith('.json') \
        else cv2.VideoWriter(args.out_file, fourcc, fps, frame_size)
    prog_bar = mmengine.ProgressBar(num_frames)
    backup_frames = []

    while ind < num_frames:
        ind += 1
        prog_bar.update()
        ret, frame = cap.read()
        if frame is None:
            # drop it when encounting None
            continue
        backup_frames.append(np.array(frame)[:, :, ::-1])
        if ind == args.sample_length:
            # provide a quick show at the beginning
            frame_queue.extend(backup_frames)
            backup_frames = []
        elif ((len(backup_frames) == args.input_step
               and ind > args.sample_length) or ind == num_frames):
            # pick a frame from the backup
            # when the backup is full or reach the last frame
            chosen_frame = backup_frames[0]
            backup_frames = []
            frame_queue.append(chosen_frame)

        ret, scores = inference(model, data, args, frame_queue)

        if ret:
            num_selected_labels = min(len(label), 5)
            scores_tuples = tuple(zip(label, scores))
            scores_sorted = sorted(
                scores_tuples, key=itemgetter(1), reverse=True)
            results = scores_sorted[:num_selected_labels]
            result_queue.append(results)

        if args.out_file.endswith('.json'):
            text_info, out_json = get_results_json(result_queue, text_info,
                                                   args.threshold, msg, ind,
                                                   out_json)
        else:
            text_info = show_results_video(result_queue, text_info,
                                           args.threshold, msg, frame,
                                           video_writer, args.label_color,
                                           args.msg_color)

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    if args.out_file.endswith('.json'):
        with open(args.out_file, 'w') as js:
            json.dump(out_json, js)
        run_action_formatter(args.out_file, args.action_out_file)


def inference(model, data, args, frame_queue):
    if len(frame_queue) != args.sample_length:
        # Do no inference when there is no enough frames
        return False, None

    cur_windows = list(np.array(frame_queue))
    if data['img_shape'] is None:
        data['img_shape'] = frame_queue[0].shape[:2]
    resized_windows = [cv2.resize(frame, (224, 224)) for frame in cur_windows]
    cur_data = data.copy()
    cur_data.update(
        dict(
            array=resized_windows,
            modality='RGB',
            frame_inds=np.arange(args.sample_length)))

    result = inference_recognizer(
        model, cur_data, test_pipeline=args.test_pipeline)
    scores = result.pred_score.tolist()

    if getattr(args, 'bg_scale', 1.0) != 1.0:
        scores = list(scores)
        scores[args.bg_index] *= args.bg_scale
        total = sum(scores)
        scores = [s / total for s in scores]

    if args.stride > 0:
        pred_stride = int(args.sample_length * args.stride)
        for _ in range(pred_stride):
            frame_queue.popleft()

    # for case ``args.stride=0``
    # deque will automatically popleft one element

    return True, scores


def main():
    args = parse_args()

    if args.out_file is None or args.out_file.endswith('.json'):
        args.out_file, args.action_out_file = resolve_json_output_paths(
            args.video_path, args.out_file)
    else:
        args.action_out_file = None

    args.device = torch.device(args.device)

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    model = init_recognizer(cfg, args.checkpoint, device=args.device)
    data = dict(img_shape=None, modality='RGB', label=-1)
    with open(args.label, 'r') as f:
        label = [line.strip() for line in f]

    # prepare test pipeline from non-camera pipeline
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
            # remove step to decode frames
            pipeline_.remove(step)
    pipeline_.insert(1, dict(type='ArrayDecode'))
    test_pipeline = Compose(pipeline_)

    assert sample_length > 0
    args.sample_length = sample_length
    args.test_pipeline = test_pipeline

    show_results(model, data, label, args)


if __name__ == '__main__':
    main()
