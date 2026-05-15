# YOLOv8n-seg Fine-Tuned Baseline Results

## Role in benchmark

YOLOv8n-seg is included as a supervised fine-tuned segmentation baseline. It is not a zero-shot foundation model. It is used to compare promptable foundation models against a small-data trained detector/segmenter.

## Dataset

Prepared YOLO segmentation dataset:

- `data/yolo_cogar_sim_500_final/`
- train images: 120
- val images: 75
- test images: 75
- labels written: 2410
- masks skipped: 0

Classes:

0. box
1. cable
2. connector
3. glass_object
4. metal_part
5. plastic_object
6. robot_gripper
7. screw
8. tool

## Training setup

- Model: `yolov8n-seg.pt`
- Epochs: 30
- Image size: 640
- Batch size: 4
- GPU: NVIDIA GTX 1050 4 GB
- Training time: 0.118 hours
- Best checkpoint:

```text
runs/segment/outputs/baselines/yolov8seg/yolov8n_seg_cogar_small/weights/best.pt
```

## Test results

Test split:

- test images: 75
- test instances: 679

Detection metrics:

| Metric | Value |
|---|---:|
| Box precision | 0.756 |
| Box recall | 0.787 |
| Box mAP50 | 0.814 |
| Box mAP50-95 | 0.679 |

Segmentation metrics:

| Metric | Value |
|---|---:|
| Mask precision | 0.761 |
| Mask recall | 0.783 |
| Mask mAP50 | 0.806 |
| Mask mAP50-95 | 0.601 |

Speed:

| Component | Time |
|---|---:|
| Preprocess | 3.4 ms |
| Inference | 11.1 ms |
| Postprocess | 12.3 ms |
| Total | 26.8 ms/image |
| Total FPS | ~37.3 FPS |
| Inference-only FPS | ~90.1 FPS |

## Per-class mask mAP50-95

| Class | Mask mAP50-95 |
|---|---:|
| box | 0.865 |
| cable | 0.463 |
| connector | 0.630 |
| glass_object | 0.575 |
| metal_part | 0.602 |
| plastic_object | 0.742 |
| robot_gripper | 0.624 |
| screw | 0.377 |
| tool | 0.528 |

## Interpretation

YOLOv8n-seg satisfies the assignment requirement for a fine-tuned baseline with mask AP reporting.

Main observations:

- Strongest classes: box, plastic_object, connector, robot_gripper.
- Hardest classes: screw, cable, tool, glass_object.
- Compared with SAM-style box-prompt models, YOLOv8n-seg is faster as an automatic feed-forward model but depends on supervised training data.
- It is useful as the practical deployment baseline when prompts are unavailable.