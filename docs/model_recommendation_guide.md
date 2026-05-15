# Model Recommendation Guide for Robotic Scene Segmentation

## Dataset and benchmark context

This guide summarizes model recommendations from the final COGAR-SimRobotics-500 benchmark.

- Dataset: COGAR-SimRobotics-500
- Images: 500
- Object instances: 4,471
- Main compared runs:
  - SAM ViT-B box
  - MobileSAM box
  - FastSAM-S box
- Additional runs:
  - SAM ViT-B point
  - SAM ViT-B automatic masks
  - SAM ViT-H CPU subset proof run

## Overall full-dataset box-prompt results

| Model | Objects | Mean IoU | Median IoU | Boundary F1 | IoU >= 0.90 | IoU < 0.10 | Mean FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAM ViT-B box | 4,471 | 0.9057 | 0.9553 | 0.9356 | 0.7513 | 0.0031 | 61.96 |
| MobileSAM box | 4,471 | 0.8656 | 0.9363 | 0.9797 | 0.6285 | 0.0045 | 69.52 |
| FastSAM-S box | 4,471 | 0.6986 | 0.8135 | 0.8920 | 0.2841 | 0.0830 | 471.30 |

## Recommended model by use case

| Use case | Recommended model | Reason |
|---|---|---|
| Highest segmentation accuracy | SAM ViT-B box | Best mean IoU, median IoU, and lowest catastrophic failure rate |
| Robotic manipulation with reliable masks | SAM ViT-B box | Most robust across object categories and challenge types |
| Edge deployment / lightweight SAM-style model | MobileSAM box | Good accuracy while using a lightweight model design |
| Real-time approximate segmentation | FastSAM-S box | Much faster in this benchmark, but lower accuracy |
| Thin objects, cables, small parts | SAM ViT-B box | Best available option, though still difficult |
| Transparent or reflective scenes | SAM ViT-B box | Best overall robustness, but failures still occur |
| Low-compute laptop or embedded system | MobileSAM box | Better balance than FastSAM-S when mask quality still matters |
| Speed-first perception pipeline | FastSAM-S box | Suitable when approximate masks are acceptable |
| Full automatic proposal mode | SAM ViT-B automatic masks | Useful when object prompts are unavailable, but less stable than box prompts |
| Research comparison with large SAM | SAM ViT-H | Checkpoint verified, but full evaluation was hardware-limited on the available 4 GB GPU |

## Final model ranking

### Accuracy ranking

1. SAM ViT-B box
2. MobileSAM box
3. FastSAM-S box

### Speed ranking

1. FastSAM-S box
2. MobileSAM box
3. SAM ViT-B box

### Practical robotics ranking

1. SAM ViT-B box for offline or high-accuracy perception
2. MobileSAM box for edge-oriented robotic perception
3. FastSAM-S box for speed-first approximate segmentation

## Prompting recommendation

Box prompts are the most reliable prompt type in this benchmark.

For SAM ViT-B:

| Prompt type | Objects | Mean IoU | Median IoU | Boundary F1 |
|---|---:|---:|---:|---:|
| Box | 4,471 | 0.9057 | 0.9553 | 0.9356 |
| Point | 4,471 | 0.7985 | 0.9125 | 0.8131 |
| Automatic masks | 4,471 | 0.8025 | 0.9422 | 0.8381 |

Box prompts are recommended when bounding boxes are available from simulation, robot perception, or an object detector. Point prompts are simpler but less stable. Automatic masks are useful when no prompts are available, but they have more failure cases.

## Category-specific recommendations

| Category | Main difficulty | Recommended model |
|---|---|---|
| box | Large regular object, usually easy | SAM ViT-B or MobileSAM |
| cable | Thin elongated geometry | SAM ViT-B |
| connector | Small part | SAM ViT-B or MobileSAM |
| glass_object | Transparent boundaries | SAM ViT-B |
| metal_part | Reflective material | SAM ViT-B |
| plastic_object | Usually easier object | SAM ViT-B or MobileSAM |
| robot_gripper | Articulated shape, holes, contact with objects | SAM ViT-B |
| screw | Very small object | SAM ViT-B |
| tool | Thin/complex object shape | SAM ViT-B |

## Challenge-specific recommendations

| Challenge | Recommended model | Notes |
|---|---|---|
| reflective_metal | SAM ViT-B | Best mean IoU among tested models |
| transparent_glass | SAM ViT-B | Still one of the hardest challenge families |
| partial_occlusion | SAM ViT-B | Most reliable but failures remain |
| small_parts | SAM ViT-B | Best option for screws/connectors |
| dynamic_scene | SAM ViT-B | Best robustness; MobileSAM acceptable for lightweight use |

## Failure-mode summary

The qualitative failure analysis shows that the hardest cases are:

- Robot grippers
- Cables and thin structures
- Small screws/connectors
- Reflective metal parts
- Transparent glass objects
- Partial occlusions
- Dynamic clutter around the robot

Robot grippers dominate the worst-case examples for SAM ViT-B and MobileSAM. FastSAM-S has more severe failures on metal parts, tools, screws, and reflective/small-part scenes.

## Hardware recommendation

SAM ViT-H was verified successfully, but full evaluation failed on the available 4 GB GPU due to CUDA out-of-memory. A small CPU subset completed successfully, so the checkpoint and implementation are valid, but the model is not practical for the available hardware.

For this project hardware:

| Model | Practical status |
|---|---|
| SAM ViT-B | Practical full benchmark model |
| SAM ViT-H | Valid but hardware-limited |
| MobileSAM | Practical lightweight model |
| FastSAM-S | Practical high-speed model |

## Final recommendation

For the final project conclusion:

- Use SAM ViT-B box prompts when segmentation accuracy is the priority.
- Use MobileSAM when a lightweight SAM-style model is needed for edge deployment.
- Use FastSAM-S when speed is more important than accurate masks.
- Avoid relying only on automatic masks for precise robotic manipulation.
- Treat transparent objects, reflective objects, cables, small parts, grippers, and occlusions as high-risk failure cases.

## EfficientSAM-Ti box-prompt recommendation

EfficientSAM-Ti is a lightweight SAM-style zero-shot segmentation model evaluated with box prompts.

Measured result on the final dataset:

- Mean IoU: 0.880745
- Median IoU: 0.939880
- Mean boundary F1: 0.910907
- IoU >= 0.90: 0.674346
- IoU >= 0.75: 0.880787
- IoU >= 0.50: 0.957057
- IoU < 0.10: 0.005592
- Mean FPS: 9.474405
- Device: CUDA, NVIDIA GTX 1050 4 GB

Recommended use:

- Use EfficientSAM-Ti when a lightweight promptable SAM-style model is needed and accuracy is more important than maximum speed.
- Use MobileSAM when the best lightweight SAM-style speed/accuracy trade-off is needed.
- Use SAM ViT-B box when maximum promptable segmentation accuracy is the priority.
- Use FastSAM-S when speed is the dominant requirement.
- Use YOLOv8n-seg when automatic supervised segmentation is allowed and prompts are unavailable.

Observed hard categories:

- cable
- robot_gripper
- tool
- screw

Observed hard challenges:

- partial_occlusion
- transparent_glass
- dynamic_scene

Final recommendation:

EfficientSAM-Ti is a valuable lightweight SAM-family benchmark. However, on this hardware and implementation, MobileSAM remains the stronger lightweight deployment recommendation because it achieves higher throughput while staying close in accuracy.

## SAM2 future-work recommendation

SAM2 was not included in the final full-object benchmark because the available local GPU had limited memory and the final evaluation focused on static simulated images.

Recommended future use:

- Evaluate SAM2 on dynamic robotic scenes or video-style frame sequences.
- Use SAM2 when temporal consistency, object persistence, or video segmentation is important.
- Compare SAM2 against SAM ViT-B box and EfficientSAM-Ti box on the same image prompts if enough GPU memory is available.

Current recommendation status:

- SAM2 is not part of the final quantitative comparison.
- It should be reported as future work, not as a completed result.
