# Benchmark Plan

COGAR I2 benchmarks zero-shot segmentation for robotic scenes on
COGAR-SimRobotics-500. The goal is to compare promptability, automatic mask
quality, runtime, and failure modes across foundation segmentation models.

## Model Families

- SAM ViT-B
- SAM ViT-H
- SAM2
- FastSAM
- MobileSAM
- EfficientSAM

## Prompt and Mask Modes

- Box prompts from object bounding boxes
- Point prompts from positive object points
- Automatic masks without per-object prompts
- Later: box plus point prompts and multiple-point strategies

## Metrics

- IoU
- mIoU
- Boundary F1
- Mask AP
- Per-category IoU
- FPS

## Failure Analysis

Report aggregate metrics and stratified failures by challenge type:

- reflective metal
- transparent glass
- partial occlusion
- small parts
- dynamic scene proxy

For each challenge, track best and worst examples, per-category IoU, confidence
calibration against ground-truth IoU, and qualitative visualizations.

## Datasets

COGAR-SimRobotics-500 is the final controlled synthetic benchmark with balanced
challenge categories and COCO annotations generated through the simulation
pipeline. The earlier real-world prototype is retained only as historical
sanity-check context.

## Implementation Direction

Keep reusable benchmark code in `src/cogar_seg/` and keep `scripts/` as thin CLI
wrappers. Model-specific adapters should live under `src/cogar_seg/models/`;
prompt builders under `src/cogar_seg/prompts/`; evaluation workflows under
`src/cogar_seg/evaluation/`; and metrics under `src/cogar_seg/metrics/`.
