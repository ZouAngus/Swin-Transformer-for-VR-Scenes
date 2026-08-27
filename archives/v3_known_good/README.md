# v3 known-good snapshot

Backed up: 2026-05-15 16:31 GMT+8

## Status
- val acc: 100%
- VR background sinking shortcut FIXED (no false squat on 0:08 / 0:17)
- 1:06 squat in 2026-05-06-17-43-00.mp4: correctly detected
- 1:55 jump #1 + 1:58 landing squat: correctly detected
- KNOWN ISSUE: 2:25-2:27 fast shallow squat in 2026-05-15-10-30-20.mp4 misclassified as jumping
  - raw probs at 2:26: Jp=0.37-0.57, Sq=0.16-0.25

## Files
- v3_best_acc_top1_epoch_2.pth (best ckpt, val acc 100%)
- my_swin_4cls_v3.py (training config)
- train_list_v3.txt (317 samples, includes 100 Move_Controller as Stand)
- val_list_v3.txt (80 samples)

## Inference recipe (k1_t04 = current best raw)
- input-step 2, stride 0.25
- smooth-k 1, decision-threshold 0.40, hysteresis 1
- stand-index 0
