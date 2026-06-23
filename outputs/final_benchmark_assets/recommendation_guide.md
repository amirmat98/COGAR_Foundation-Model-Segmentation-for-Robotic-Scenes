# Final Recommendation Guide

> **Legacy results:** this generated guide combines the original full-dataset
> zero-shot run with 50-image baseline validation results. It is retained for
> provenance but must not be used as the final model comparison. Regenerate it
> after the common held-out test summaries are complete.

This guide summarizes which segmentation model to use under the benchmark conditions. It uses the compact Task 6, Task 7, Task 8, and Task 9 outputs.

## High-Level Recommendations

- Use box prompts when a detector, tracker, or robot prior can provide a target region. Box prompting consistently dominates point prompting for quality.
- Use SAM2 or SAM ViT-H/B when segmentation quality is the main goal and real-time operation is not required.
- Use YOLOv8-seg or DeepLabV3+ when GPU real-time throughput is required. They are supervised baselines, so they depend on labeled data from the target domain.
- Use MobileSAM for the best lightweight point/box speed-size trade-off. It is much smaller than SAM ViT-H/B and faster than EfficientSAM-S in this benchmark.
- Use EfficientSAM-S when lightweight box-prompt quality matters more than speed. Its box-prompt mIoU is strong, but it is slower and larger than MobileSAM.
- Avoid automatic mask generation for real-time robot loops in this setup. Automatic modes are much slower, and EfficientSAM grid automatic quality is weak.
- Treat reflective/transparent objects, small/thin parts, and occluded robot-body regions as the highest-risk qualitative failure areas.

## Plots

### Dataset examples

![Dataset examples](plots/dataset_examples.png)

### Zero-shot mIoU heatmap

![Zero-shot mIoU heatmap](plots/zero_shot_miou_heatmap.png)

### Baseline mIoU bars

![Baseline mIoU bars](plots/baseline_miou_bars.png)

### CUDA speed-quality scatter

![CUDA speed-quality scatter](plots/cuda_speed_quality_scatter.png)

### Lightweight SAM trade-off

![Lightweight SAM trade-off](plots/lightweight_sam_tradeoff_cuda.png)

### Challenge group summary

![Challenge group summary](plots/challenge_group_weighted_iou.png)

### Zero-shot winners

![Zero-shot winners](plots/zero_shot_dataset_prompt_winners.png)


## Best Overall CUDA Quality by Dataset

| dataset | model | prompt_mode | mIoU | fps | mask_AP |
| --- | --- | --- | --- | --- | --- |
| BlenderProc | SAM ViT-H | Box | 0.923 | 0.574 | 0.868 |
| Isaac G1 | SAM ViT-H | Box | 0.752 | 0.574 | 0.678 |
| OCID | DeepLabV3+ | Inference | 0.963 | 37.811 | N/A |

## Best CUDA Speed-Quality Product by Dataset

| dataset | model | prompt_mode | mIoU | fps | miou_fps_product |
| --- | --- | --- | --- | --- | --- |
| BlenderProc | YOLOv8-seg | Inference | 0.861 | 41.647 | 35.866 |
| Isaac G1 | DeepLabV3+ | Inference | 0.660 | 34.332 | 22.675 |
| OCID | DeepLabV3+ | Inference | 0.963 | 37.811 | 36.421 |

## Best Lightweight CUDA Trade-Off by Dataset and Prompt

| dataset | prompt_mode | model | mIoU | fps | checkpoint_size_mb | miou_fps_product |
| --- | --- | --- | --- | --- | --- | --- |
| BlenderProc | Automatic | EffSAM-Ti | 0.120 | 2.517 | 40.980 | 0.303 |
| BlenderProc | Box | MobileSAM | 0.883 | 15.737 | 40.730 | 13.891 |
| BlenderProc | Point | MobileSAM | 0.740 | 15.618 | 40.730 | 11.556 |
| Isaac G1 | Automatic | MobileSAM | 0.421 | 0.217 | 40.730 | 0.092 |
| Isaac G1 | Box | MobileSAM | 0.693 | 16.895 | 40.730 | 11.701 |
| Isaac G1 | Point | MobileSAM | 0.516 | 16.836 | 40.730 | 8.694 |
| OCID | Automatic | EffSAM-S | 0.141 | 1.934 | 105.740 | 0.272 |
| OCID | Box | MobileSAM | 0.824 | 15.522 | 40.730 | 12.786 |
| OCID | Point | MobileSAM | 0.674 | 15.557 | 40.730 | 10.484 |

## Best Supervised Baseline by Dataset

| dataset | baseline | evaluation_type | mIoU | boundary_f1 | mask_AP |
| --- | --- | --- | --- | --- | --- |
| BlenderProc | YOLOv8-seg | instance_segmentation | 0.861 | 0.814 | 0.643 |
| Isaac G1 | DeepLabV3+ | semantic_segmentation | 0.660 | 0.664 | N/A |
| OCID | DeepLabV3+ | semantic_segmentation | 0.963 | 0.880 | N/A |

## Best Zero-Shot Prompted Model by Dataset and Prompt

| dataset | prompt_mode | model | mIoU | boundary_f1 | mask_AP |
| --- | --- | --- | --- | --- | --- |
| BlenderProc | Automatic | SAM ViT-H | 0.878 | 0.855 | 0.581 |
| BlenderProc | Box | SAM ViT-H | 0.923 | 0.905 | 0.868 |
| BlenderProc | Point | SAM2-L | 0.827 | 0.822 | 0.751 |
| Isaac G1 | Automatic | FastSAM-X | 0.523 | 0.600 | 0.256 |
| Isaac G1 | Box | SAM ViT-H | 0.752 | 0.874 | 0.678 |
| Isaac G1 | Point | EffSAM-Ti | 0.603 | 0.688 | 0.510 |
| OCID | Automatic | SAM ViT-H | 0.838 | 0.776 | 0.396 |
| OCID | Box | SAM2-L | 0.866 | 0.787 | 0.776 |
| OCID | Point | SAM2-L | 0.759 | 0.737 | 0.612 |

## Lowest Challenge-Group Rows

| run_type | dataset | model | prompt_mode | challenge_group | weighted_iou | mean_boundary_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| zero_shot | Isaac G1 | FastSAM-X | Point | robot_and_occlusion | 0.174 | 0.576 |
| zero_shot | Isaac G1 | SAM2-L | Automatic | robot_and_occlusion | 0.181 | 0.611 |
| zero_shot | BlenderProc | FastSAM-X | Point | robot_and_occlusion | 0.192 | 0.228 |
| zero_shot | Isaac G1 | SAM ViT-B | Automatic | robot_and_occlusion | 0.209 | 0.629 |
| zero_shot | BlenderProc | FastSAM-X | Point | small_parts_thin_structures | 0.245 | 0.299 |
| zero_shot | Isaac G1 | FastSAM-X | Box | robot_and_occlusion | 0.269 | 0.690 |
| zero_shot | Isaac G1 | SAM ViT-H | Automatic | robot_and_occlusion | 0.276 | 0.681 |
| zero_shot | Isaac G1 | FastSAM-X | Automatic | robot_and_occlusion | 0.278 | 0.703 |

## Mask R-CNN Implementation Note

Mask R-CNN is implemented with TorchVision's `maskrcnn_resnet50_fpn` instead of Detectron2. This keeps the baseline reproducible in the existing PyTorch environment while still evaluating the requested Mask R-CNN baseline family. A Detectron2 run would be a duplicate Mask R-CNN implementation rather than a different baseline category.
