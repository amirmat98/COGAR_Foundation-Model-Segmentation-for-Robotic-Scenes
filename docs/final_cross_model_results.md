# Final Cross-Model Results on COGAR-SimRobotics-500

## Benchmark scope

All full-dataset zero-shot results use the same final COGAR-SimRobotics-500
index with 500 images and 4,471 object instances.

This page summarizes cross-model results across:

- SAM ViT-B
- SAM2.1-Tiny
- FastSAM-S
- MobileSAM
- EfficientSAM-Ti
- SAM ViT-H CPU subset reference
- YOLOv8n-seg supervised baseline
- Mask R-CNN supervised baseline

## Full-dataset promptable and lightweight comparison

| Model / mode | Objects | Mean IoU | Median IoU | Boundary F1 | IoU >= 0.90 | IoU >= 0.75 | IoU >= 0.50 | IoU < 0.10 | Mean FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM2.1-Tiny box | 4,471 | 0.912746 | 0.955280 | 0.930659 | 0.754641 | 0.926191 | 0.981660 | 0.000895 | 16.809629 |
| SAM ViT-B box | 4,471 | 0.9057 | 0.9553 | 0.9356 | 0.7513 | 0.9137 | 0.9703 | 0.0031 | 61.96 |
| EfficientSAM-Ti box | 4,471 | 0.880745 | 0.939880 | 0.910907 | 0.674346 | 0.880787 | 0.957057 | 0.005592 | 9.474405 |
| MobileSAM box | 4,471 | 0.8656 | 0.9363 | 0.9797 | 0.6285 | 0.8430 | 0.9457 | 0.0045 | 69.52 |
| FastSAM-S box | 4,471 | 0.698569 | 0.813478 | 0.891956 | 0.284053 | 0.592709 | 0.808544 | 0.082979 | 471.295280 |

## Prompt-mode comparison

| Model | Prompt mode | Objects | Mean IoU | Median IoU | Boundary F1 | IoU >= 0.75 | IoU < 0.10 | Mean FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SAM ViT-B | Box | 4,471 | 0.9057 | 0.9553 | 0.9356 | 0.9137 | 0.0031 | 61.96 |
| SAM ViT-B | Point | 4,471 | 0.7985 | 0.9125 | 0.8131 | 0.7271 | 0.0204 | not evaluated |
| SAM ViT-B | Auto | 4,471 | 0.8025 | 0.9422 | 0.8381 | 0.7486 | 0.0552 | not evaluated |
| SAM2.1-Tiny | Box | 4,471 | 0.912746 | 0.955280 | 0.930659 | 0.926191 | 0.000895 | 16.809629 |
| SAM2.1-Tiny | Point | 4,471 | 0.865783 | 0.934924 | 0.873056 | 0.827555 | 0.004921 | 16.679109 |
| SAM2.1-Tiny | Auto | 4,471 | 0.640259 | 0.870136 | 0.678148 | 0.586670 | 0.224782 | 2.300666 |
| FastSAM-S | Box | 4,471 | 0.698569 | 0.813478 | 0.891956 | 0.592709 | 0.082979 | 471.295280 |
| FastSAM-S | Point | 4,471 | 0.759372 | 0.888325 | 0.788963 | 0.710356 | 0.073809 | 214.069901 |
| FastSAM-S | Auto / Everything | 4,471 | 0.777331 | 0.891437 | 0.809290 | 0.720197 | 0.050772 | 206.475290 |

## SAM ViT-H CPU subset reference

| Prompt mode | Subset | Device | Mean IoU | Median IoU | Boundary F1 | Mean FPS |
|---|---:|---|---:|---:|---:|---:|
| Box | 25 objects | CPU | 0.9449 | 0.9717 | 0.9637 | 0.1820 |
| Point | 25 objects | CPU | 0.7721 | 0.9547 | 0.7958 | 0.1762 |
| Auto | 42 objects / 5 images | CPU | 0.7302 | 0.9563 | 0.7640 | 0.2118 |

SAM ViT-H is excluded from the full cross-model ranking because full CUDA
evaluation is infeasible on the available GTX 1050 4 GB GPU.

## Per-category mean IoU for main box-prompt runs

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

## Per-challenge mean IoU for main box-prompt runs

| Challenge | FastSAM-S box | MobileSAM box | SAM ViT-B box |
|---|---:|---:|---:|
| dynamic_scene | 0.7071 | 0.8669 | 0.9091 |
| partial_occlusion | 0.6889 | 0.8590 | 0.8964 |
| reflective_metal | 0.7180 | 0.8746 | 0.9138 |
| small_parts | 0.6942 | 0.8660 | 0.9120 |
| transparent_glass | 0.6895 | 0.8624 | 0.8946 |

Detailed per-category and per-challenge tables for SAM2.1-Tiny and
EfficientSAM-Ti are reported in their dedicated result pages.

## Supervised baseline comparison

| Model | Evaluation split | Main result | Speed |
|---|---|---|---|
| YOLOv8n-seg | 75 test images / 679 instances | mask precision 0.761, mask recall 0.783, mask mAP50 0.806, mask mAP50-95 0.601 | 26.8 ms/image, about 37.3 FPS total |
| Mask R-CNN ResNet-50 FPN | 75 test images / 679 objects | mean IoU 0.7462, median IoU 0.8309, boundary F1 0.7218 | 5.5855 image FPS |

YOLOv8n-seg and Mask R-CNN are supervised baselines, not zero-shot models.
DeepLabV3+ is excluded because it is a semantic segmentation model and is not
directly comparable under the instance-mask protocol used here.

## Interpretation

SAM2.1-Tiny box gives the highest completed full-dataset promptable mean IoU.
SAM ViT-B box remains the stronger speed/accuracy balance in the current
implementation. MobileSAM is the best lightweight SAM-style edge trade-off.
EfficientSAM-Ti has strong IoU but slower measured FPS. FastSAM-S is the
speed-first zero-shot model, with substantially lower box-prompt mask quality.

Across object categories, cables, robot grippers, screws, and tools are the
hardest object types. These include thin structures, small parts, articulated
shapes, and ambiguous boundaries.

Across challenge groups, transparent glass and partial occlusion remain hard for
the strongest prompted models, while automatic-mask modes are especially weak on
small parts and missed proposals.

## Speed note

The reported FPS values for promptable models are object-row-level timing values
from the evaluation scripts. Because several objects can come from the same image
and image-level embeddings may be reused, these values should be treated as
relative speed indicators rather than strict robot camera-frame rates.
