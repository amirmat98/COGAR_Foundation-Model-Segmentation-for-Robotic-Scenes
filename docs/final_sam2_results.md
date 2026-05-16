# SAM2.1-Tiny Zero-Shot Results

## Role in benchmark

SAM2.1-Tiny is included as the SAM2-family zero-shot segmentation model in the benchmark.

It was evaluated on the final simulated robotic-scene dataset:

- `data/cogar_sim_500_final/annotations/sim_robotic_scenes_index_final_filtered.csv`

SAM2.1-Tiny was installed in a separate environment under:

- `/mnt/Info/COGAR_Large/SAM2/`

This was done to avoid modifying the main benchmark environment.

## Evaluation setup

- Model: SAM2.1-Tiny / SAM2.1 Hiera Tiny
- Checkpoint: `sam2.1_hiera_tiny.pt`
- Prompt modes evaluated: box, point, and automatic mask generation
- Device: CUDA
- GPU: NVIDIA GTX 1050 4 GB
- Objects evaluated: 4,471
- Image dataset: 500 simulated robotic-scene images

## Prompt protocol

### Box prompt

Box prompts use the ground-truth object bounding box:

- `bbox_xmin`
- `bbox_ymin`
- `bbox_xmax`
- `bbox_ymax`

### Point prompt

Point prompts use the stored object point:

- `point_x`
- `point_y`

For each prompt, SAM2 predicts multiple masks. The benchmark selects the mask with the highest IoU against the ground-truth object mask for object-level scoring.

### Automatic mask generation

Automatic mask generation is prompt-free. SAM2 samples points over an image grid,
predicts candidate masks, filters masks, and the benchmark matches generated
masks to ground-truth object instances for scoring.

## Overall prompted results

| Prompt type | Objects | Mean IoU | Median IoU | Mean boundary F1 | IoU >= 0.90 | IoU >= 0.75 | IoU >= 0.50 | IoU < 0.10 | Mean predicted IoU | Mean FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Box | 4,471 | 0.912746 | 0.955280 | 0.930659 | 0.754641 | 0.926191 | 0.981660 | 0.000895 | 0.935720 | 16.809629 |
| Point | 4,471 | 0.865783 | 0.934924 | 0.873056 | 0.614404 | 0.827555 | 0.952807 | 0.004921 | 0.804797 | 16.679109 |

## Mean IoU by category: box prompt

| Category | Count | Mean IoU | Median IoU | Mean boundary F1 | IoU >= 0.75 | IoU < 0.10 |
|---|---:|---:|---:|---:|---:|---:|
| cable | 281 | 0.848384 | 0.895760 | 0.905199 | 0.829181 | 0.007117 |
| robot_gripper | 1042 | 0.859480 | 0.912873 | 0.848819 | 0.849328 | 0.000960 |
| tool | 296 | 0.880302 | 0.935766 | 0.889262 | 0.878378 | 0.003378 |
| screw | 427 | 0.897325 | 0.919831 | 0.988927 | 0.946136 | 0.000000 |
| glass_object | 360 | 0.934410 | 0.966264 | 0.895568 | 0.952778 | 0.000000 |
| connector | 531 | 0.938521 | 0.959478 | 0.985101 | 0.971751 | 0.000000 |
| metal_part | 555 | 0.952624 | 0.972770 | 0.974476 | 0.978378 | 0.000000 |
| box | 352 | 0.955106 | 0.979920 | 0.945244 | 0.977273 | 0.000000 |
| plastic_object | 627 | 0.962588 | 0.979200 | 0.985004 | 0.977671 | 0.000000 |

## Mean IoU by category: point prompt

| Category | Count | Mean IoU | Median IoU | Mean boundary F1 | IoU >= 0.75 | IoU < 0.10 |
|---|---:|---:|---:|---:|---:|---:|
| robot_gripper | 1042 | 0.767225 | 0.827355 | 0.749527 | 0.641075 | 0.014395 |
| cable | 281 | 0.800109 | 0.842007 | 0.860588 | 0.697509 | 0.000000 |
| tool | 296 | 0.839987 | 0.911390 | 0.845837 | 0.773649 | 0.000000 |
| glass_object | 360 | 0.850406 | 0.918180 | 0.757417 | 0.777778 | 0.002778 |
| screw | 427 | 0.873286 | 0.905063 | 0.966363 | 0.913349 | 0.002342 |
| connector | 531 | 0.917311 | 0.953146 | 0.955066 | 0.937853 | 0.001883 |
| metal_part | 555 | 0.918042 | 0.965753 | 0.928011 | 0.915315 | 0.003604 |
| box | 352 | 0.934325 | 0.976164 | 0.910163 | 0.937500 | 0.002841 |
| plastic_object | 627 | 0.946527 | 0.975720 | 0.960708 | 0.958533 | 0.001595 |

## Interpretation

SAM2.1-Tiny performs very strongly on the simulated robotic-scene dataset.

Main observations:

- SAM2.1-Tiny box prompting achieved the highest mean IoU among the completed promptable full-dataset benchmarks.
- SAM2.1-Tiny box slightly outperformed SAM ViT-B box in mean IoU.
- SAM2.1-Tiny point prompting also performed strongly, but lower than box prompting.
- The hardest categories for SAM2.1-Tiny box are cable, robot_gripper, tool, and screw.
- The strongest categories for SAM2.1-Tiny box are plastic_object, box, metal_part, and connector.
- SAM2.1-Tiny is slower than SAM ViT-B and much slower than FastSAM-S in the current implementation, but it is accurate and feasible on the GTX 1050 4 GB with the tiny checkpoint.

## Limitations

SAM2 was installed in a separate environment on `/mnt/Info/COGAR_Large/SAM2/` because the official SAM2 dependency stack required a newer PyTorch/torchvision version than the main benchmark environment.

Generated result files may remain local depending on size and reproducibility
needs. The final metrics are summarized in this document.

- `outputs/tables/sam2/final_box_cuda/`
- `outputs/tables/sam2/final_point_cuda/`

## Conclusion

SAM2.1-Tiny is included as a completed SAM2-family zero-shot model in the benchmark. Box, point, and automatic mask generation were evaluated on all 4,471 object instances.

## SAM2.1-Tiny automatic mask generation

SAM2.1-Tiny automatic mask generation was evaluated on the full 500-image dataset using conservative settings suitable for the GTX 1050 4 GB GPU:

- `points_per_side`: 16
- `pred_iou_thresh`: 0.80
- `stability_score_thresh`: 0.90
- `crop_n_layers`: 0
- Device: CUDA

Automatic mask generation is prompt-free. SAM2 samples point prompts over an image grid, predicts masks, filters masks, and then the benchmark matches generated masks to ground-truth object instances for scoring.

### Overall auto results

| Metric | Value |
|---|---:|
| Objects | 4,471 |
| Mean IoU | 0.640259 |
| Median IoU | 0.870136 |
| Mean boundary F1 | 0.678148 |
| IoU >= 0.90 | 0.470588 |
| IoU >= 0.75 | 0.586670 |
| IoU >= 0.50 | 0.683292 |
| IoU < 0.10 | 0.224782 |
| Mean predicted IoU | 0.940974 |
| Mean FPS | 2.300666 |
| Total model time | 1943.350 s |

### Mean IoU by category: auto

| Category | Count | Mean IoU | Median IoU | Mean boundary F1 | IoU >= 0.75 | IoU < 0.10 |
|---|---:|---:|---:|---:|---:|---:|
| screw | 427 | 0.262570 | 0.000876 | 0.323649 | 0.238876 | 0.676815 |
| robot_gripper | 1042 | 0.508220 | 0.613787 | 0.557955 | 0.384837 | 0.245681 |
| cable | 281 | 0.540056 | 0.685232 | 0.604072 | 0.427046 | 0.266904 |
| glass_object | 360 | 0.603488 | 0.787498 | 0.605631 | 0.527778 | 0.216667 |
| tool | 296 | 0.663334 | 0.851463 | 0.691366 | 0.570946 | 0.179054 |
| connector | 531 | 0.737014 | 0.942533 | 0.783449 | 0.738230 | 0.169492 |
| metal_part | 555 | 0.773017 | 0.964469 | 0.812575 | 0.760360 | 0.153153 |
| plastic_object | 627 | 0.844045 | 0.975248 | 0.870719 | 0.848485 | 0.102073 |
| box | 352 | 0.869217 | 0.976169 | 0.872345 | 0.838068 | 0.042614 |

### Mean IoU by challenge: auto

| Challenge | Count | Mean IoU | Median IoU | Mean boundary F1 | IoU >= 0.75 | IoU < 0.10 |
|---|---:|---:|---:|---:|---:|---:|
| small_parts | 1551 | 0.586400 | 0.830841 | 0.635217 | 0.554481 | 0.301096 |
| partial_occlusion | 3187 | 0.627681 | 0.830764 | 0.665380 | 0.566050 | 0.225290 |
| transparent_glass | 830 | 0.629659 | 0.848286 | 0.651270 | 0.563855 | 0.203614 |
| dynamic_scene | 797 | 0.681483 | 0.909036 | 0.721684 | 0.633626 | 0.191970 |
| reflective_metal | 1161 | 0.712922 | 0.932247 | 0.747000 | 0.663221 | 0.168820 |

### Auto interpretation

SAM2.1-Tiny automatic mask generation is substantially weaker than SAM2.1-Tiny box and point prompting.

Main observations:

- Auto mode is prompt-free and therefore much harder than box or point prompting.
- Screws are the hardest category, with very low mean IoU and many failures.
- Small parts are the hardest challenge group.
- Reflective metal is the strongest challenge group in auto mode.
- Auto mode is slow on GTX 1050, with approximately 2.30 FPS measured over object-instance scoring.

This completes SAM2.1-Tiny coverage for box, point, and automatic mask generation.
