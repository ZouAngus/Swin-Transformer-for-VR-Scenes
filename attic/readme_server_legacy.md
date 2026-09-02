# MMAction2 Custom Video Training & Inference Guide

---

## Environment Setup

- Directly use the environment `mmaction2` on server2

- Or follow MMAction2 installation guide:  
  [https://mmaction2.readthedocs.io/en/latest/get_started/installation.html]

- Compatible package versions for our server environment:

| Package | Version          | Install Command                                                                                  |
|---------|------------------|------------------------------------------------------------------------------------------------|
| mmcv    | 2.0.1            | `pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html`|
| torch   | 1.11.0 + cu113   | `pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113` |



---

## Dataset Overview

- Our dataset contains **18 classes**:  
  - 7 normal actions  
  - 11 VR actions

- VR actions and corresponding Excel row indices (action names consistent with `classInd.txt`):

| Scene          | Action          | Row index |
|----------------|-----------------|-----------|
| gaming museum  | Move_Controller | 21        |
|                | Picking_item    | 22        |
|                | Measure_Length  | 23        |
| bowling        | Bowling         | 21        |
| gallery        | Move_Controller | 21        |
|                | Waive_sword     | 22        |
|                | Measure_Length  | 23        |
| travel         | Move_Controller | 21        |
|                | Catching_fish   | 22        |
|                | Grabbing        | 23        |
|                | Measure_Length  | 24        |
| boss           | Waive           | 21        |
|                | Throw           | 22        |
|                | Waive_sword     | 23        |
|                | Cutting         | 24        |
|                | Shooting        | 25        |
| candy          | Shooting        | 21        |

---

## Dataset Paths

- Raw sliced videos (based on Excel) are located on our server1 at:  
  `/data/sda1/shared/sliced_video`

  - Video naming format:  
    `s{person}_a{action}_{scene}_{excel row number}_rep{repetition}_frames_{start_frame}_{end_frame}_{view}.mp4`  


- Sliced videos (final versions for training) : 
  - Training videos: `data/myvideo/videos_train`  
  - Validation videos: `data/myvideo/videos_val`

  - Long action videos have been segmented into small clips.
  - Videos under `./Stand` that end with `_bg` represent background frames (no action annotation).

- Note:  
  There is an approximate 6-frame offset between views C and L.  
  All `start_frame` and `end_frame` correspond to view C.


---

## Checkpoints

```
checkpoints

├── bg_7289/
│   └── best_acc_top1_epoch_27.pth      # trained with videos containing background frames
├── no_bg/
│   └── best_acc_top1_epoch_24.pth      # trained only with action annotated frames
└── class_7/
    └── best_acc_top1_epoch_18.pth      # action recognition for 7 general classes
```

---

## Training & Testing for Action Recognition
- Make sure the number of classes in `configs/_base_/models/swin_tiny.py` (line 25) matches the actual number of classes.
### Training

```bash
python tools/train.py configs/recognition/swin/my_swin.py --work-dir ...
```

### Testing
```bash
python tools/test.py configs/recognition/swin/my_swin.py checkpoints/bg_7289/best_acc_top1_epoch_27.pth --work-dir ... --dump ... 
```

## Action Inference on Long Videos

```bash
python demo/long_video_demo.py \
  --config demo/demo_configs/my_swin_demo.py \
  --checkpoint checkpoints/bg_7289/best_acc_top1_epoch_27.pth \
  --video_path ... \
  --label tools/my_label.txt \
  --out_file ... \
  --input-step 2 \
  --stride 0.25
```


## VSCode Video Preview Tip

- It is convenient to keep videos encoded in H.264 format for easy preview in VSCode.


