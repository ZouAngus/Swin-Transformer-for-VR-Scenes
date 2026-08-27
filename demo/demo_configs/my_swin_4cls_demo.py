# Copyright (c) OpenMMLab. All rights reserved.
# Demo config for the 4-class fine-tuned Swin-tiny recognizer.
# num_classes=4 already lives in _base_/models/swin_tiny.py.
_base_ = [
    '../../configs/_base_/models/swin_tiny.py'
]

# Disable 2D->3D inflation; weights come from --checkpoint at runtime.
model = dict(
    backbone=dict(pretrained=None, pretrained2d=False),
    cls_head=dict(num_classes=4),
)

dataset_type = 'VideoDataset'
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        ann_file=None,
        data_prefix=None,
        pipeline=test_pipeline))
