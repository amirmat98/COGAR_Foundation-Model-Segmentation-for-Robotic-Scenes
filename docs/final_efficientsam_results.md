# EfficientSAM-Ti Box-Prompt Results

## Role in benchmark

EfficientSAM-Ti is included as a lightweight SAM-style zero-shot segmentation model.

It is evaluated with box prompts on the same final simulated robotic-scene dataset used for the other foundation-model benchmarks:

- `data/cogar_sim_500_final/annotations/sim_robotic_scenes_index_final_filtered.csv`

EfficientSAM-Ti was evaluated using the official EfficientSAM-Ti checkpoint kept outside the repository:

- `~/Desktop/COGAR/external/EfficientSAM/weights/efficient_sam_vitt.pt`

## Visual evidence

![EfficientSAM-Ti in edge trade-off chart](/outputs/figures/final_report/edge_tradeoff/iou_vs_fps_tradeoff.png)

*Figure: EfficientSAM-Ti shows strong IoU but lower measured FPS than MobileSAM in the final edge trade-off chart.*

## Evaluation setup

- Model: EfficientSAM-Ti
- Prompt type: box
- Box prompt encoding: top-left corner label 2 and bottom-right corner label 3
- Dataset objects evaluated: 4471
- Device: cuda
- GPU: NVIDIA GTX 1050 4 GB
- Output folder: `outputs/tables/efficientsam/final_ti_cuda_fixed/`

## Overall results

| Metric | Value |
|---|---:|
| Mean IoU | 0.880745 |
| Median IoU | 0.939880 |
| Mean boundary F1 | 0.910907 |
| IoU >= 0.90 | 0.674346 |
| IoU >= 0.75 | 0.880787 |
| IoU >= 0.50 | 0.957057 |
| IoU < 0.10 | 0.005592 |
| Mean predicted IoU | 0.927803 |
| Mean FPS | 9.474405 |
| Total model time | 471.903 s |

## Mean IoU by category

| Category | Count | Mean IoU | Median IoU | Mean boundary F1 | IoU >= 0.75 | IoU < 0.10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cable | 281 | 0.771224 | 0.832273 | 0.835197 | 0.672598 | 0.021352 |
| robot_gripper | 1042 | 0.824218 | 0.897217 | 0.847314 | 0.797505 | 0.009597 |
| tool | 296 | 0.847494 | 0.926955 | 0.874856 | 0.827703 | 0.010135 |
| screw | 427 | 0.865641 | 0.900000 | 0.964371 | 0.901639 | 0.002342 |
| glass_object | 360 | 0.893864 | 0.954612 | 0.857211 | 0.897222 | 0.002778 |
| connector | 531 | 0.918184 | 0.950748 | 0.963063 | 0.945386 | 0.000000 |
| metal_part | 555 | 0.927033 | 0.955866 | 0.959827 | 0.954955 | 0.001802 |
| box | 352 | 0.927036 | 0.973090 | 0.919670 | 0.940341 | 0.008523 |
| plastic_object | 627 | 0.943557 | 0.970640 | 0.969568 | 0.960128 | 0.000000 |

## Mean IoU by challenge

| Challenge | Count | Mean IoU | Median IoU | Mean boundary F1 | IoU >= 0.75 | IoU < 0.10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| partial_occlusion | 3187 | 0.870699 | 0.935085 | 0.900687 | 0.865077 | 0.007531 |
| transparent_glass | 830 | 0.881158 | 0.944123 | 0.878424 | 0.879518 | 0.006024 |
| dynamic_scene | 797 | 0.881531 | 0.948617 | 0.912978 | 0.875784 | 0.010038 |
| small_parts | 1551 | 0.886124 | 0.932168 | 0.938662 | 0.901999 | 0.002579 |
| reflective_metal | 1161 | 0.893422 | 0.948571 | 0.921265 | 0.898363 | 0.003445 |

## Interpretation

EfficientSAM-Ti performs strongly as a lightweight promptable segmentation model.

Main observations:

- Overall mean IoU is 0.880745, below SAM ViT-B box but above FastSAM-S box.
- Median IoU is high at 0.939880, showing that most prompted objects are segmented accurately.
- The model is slower than MobileSAM and FastSAM-S in this implementation on GTX 1050, with 9.474405 FPS measured over object prompts.
- The hardest categories are cable, robot_gripper, tool, and screw.
- The strongest categories are plastic_object, box, metal_part, and connector.
- The hardest challenge group is partial_occlusion.
- Transparent glass and dynamic scenes are also challenging but remain above 0.88 mean IoU.

## Comparison position

Updated model positioning:

- SAM ViT-B box remains the best accuracy-oriented promptable model.
- MobileSAM box remains the best lightweight SAM-style speed/accuracy trade-off.
- EfficientSAM-Ti is a strong lightweight SAM-style accuracy baseline, but slower in this implementation.
- FastSAM-S remains the speed-first promptable baseline.
- YOLOv8n-seg remains the supervised fine-tuned automatic segmentation baseline.

## Notes

EfficientSAM-Ti is zero-shot with respect to the COGAR simulated robotic scene dataset. It was not fine-tuned on the dataset.

Generated result files are intentionally not committed:

- `outputs/tables/efficientsam/final_ti_cuda_fixed/`

Only source scripts and documentation should be committed.
