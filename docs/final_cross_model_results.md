# Final Cross-Model Results on COGAR-SimRobotics-500

## Compared models

This benchmark compares three completed full-dataset segmentation runs:

- SAM ViT-B with box prompts
- MobileSAM with box prompts
- FastSAM-S with box-style mask selection

All models were evaluated on the same final COGAR-SimRobotics-500 index with 500 images and 4,471 object instances.

## Overall results

| Model | Objects | Mean IoU | Median IoU | Boundary F1 | IoU >= 0.90 | IoU >= 0.75 | IoU >= 0.50 | IoU < 0.10 | Mean FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM ViT-B box | 4,471 | 0.9057 | 0.9553 | 0.9356 | 0.7513 | 0.9137 | 0.9703 | 0.0031 | 61.96 |
| MobileSAM box | 4,471 | 0.8656 | 0.9363 | 0.9797 | 0.6285 | 0.8430 | 0.9457 | 0.0045 | 69.52 |
| FastSAM-S box | 4,471 | 0.6986 | 0.8135 | 0.8920 | 0.2841 | 0.5927 | 0.8085 | 0.0830 | 471.30 |

## Per-category mean IoU

| Category | FastSAM-S box | MobileSAM box | SAM ViT-B box |
|---|---:|---:|---:|
| box | 0.7853 | 0.9089 | 0.9429 |
| cable | 0.4910 | 0.7395 | 0.8492 |
| connector | 0.7509 | 0.9059 | 0.9410 |
| glass_object | 0.7320 | 0.8922 | 0.9040 |
| metal_part | 0.7878 | 0.9263 | 0.9443 |
| plastic_object | 0.7740 | 0.9367 | 0.9602 |
| robot_gripper | 0.6426 | 0.8042 | 0.8462 |
| screw | 0.5978 | 0.8415 | 0.8963 |
| tool | 0.6733 | 0.8149 | 0.8890 |

## Per-challenge mean IoU

| Challenge | FastSAM-S box | MobileSAM box | SAM ViT-B box |
|---|---:|---:|---:|
| dynamic_scene | 0.7071 | 0.8669 | 0.9091 |
| partial_occlusion | 0.6889 | 0.8590 | 0.8964 |
| reflective_metal | 0.7180 | 0.8746 | 0.9138 |
| small_parts | 0.6942 | 0.8660 | 0.9120 |
| transparent_glass | 0.6895 | 0.8624 | 0.8946 |

## Interpretation

SAM ViT-B gives the best overall segmentation accuracy. It achieves the highest mean IoU, median IoU, and the lowest catastrophic failure rate. It is the best choice when reliable segmentation masks are the priority.

MobileSAM provides the strongest lightweight trade-off. It is less accurate than SAM ViT-B, but it remains close on median IoU and keeps a very low catastrophic failure rate. This makes it suitable for edge-oriented robotic perception when compute and deployment constraints matter.

FastSAM-S is the fastest model in this benchmark, but it has the lowest segmentation accuracy. Its mean IoU is much lower than SAM ViT-B and MobileSAM, and its catastrophic failure rate is much higher. It is best interpreted as a high-speed, lower-accuracy baseline.

Across object categories, cables, robot grippers, screws, and tools are the hardest object types. These categories include thin structures, small parts, articulated shapes, and ambiguous boundaries.

Across robotic-scene challenges, transparent glass and partial occlusion are the hardest for SAM ViT-B, while FastSAM-S is weaker across all challenge families.

## Speed note

The reported FPS values are object-row-level timing values from the evaluation scripts. Because several objects can come from the same image and image-level predictions or embeddings may be reused, the FPS values should be treated as relative speed indicators, not strict per-frame robot deployment FPS.

## EfficientSAM-Ti addition

EfficientSAM-Ti box-prompt evaluation was added as the lightweight EfficientSAM-family benchmark.

EfficientSAM-Ti overall results:

- Mean IoU: 0.880745
- Median IoU: 0.939880
- Mean boundary F1: 0.910907
- IoU >= 0.90: 0.674346
- IoU >= 0.75: 0.880787
- IoU >= 0.50: 0.957057
- IoU < 0.10: 0.005592
- Mean predicted IoU: 0.927803
- Mean FPS: 9.474405
- Device: CUDA, NVIDIA GTX 1050 4 GB

Category-level findings:

- Hardest categories: cable, robot_gripper, tool, screw.
- Strongest categories: plastic_object, box, metal_part, connector.

Challenge-level findings:

- Hardest challenge: partial_occlusion.
- Other difficult challenges: transparent_glass and dynamic_scene.
- Reflective metal and small-parts scenes remained comparatively stronger.

Updated interpretation:

- SAM ViT-B box remains the best accuracy-oriented promptable model.
- MobileSAM box remains the best lightweight SAM-style trade-off.
- EfficientSAM-Ti is a strong lightweight SAM-family accuracy baseline but is slower than MobileSAM and FastSAM-S in this implementation.
- FastSAM-S remains the speed-first promptable baseline.
- YOLOv8n-seg remains the supervised fine-tuned automatic segmentation baseline.
