# Assignment Task 9: Lightweight SAM Variants for Edge Deployment

## Task requirement

Test whether lightweight SAM variants such as MobileSAM and EfficientSAM can provide a good trade-off for edge deployment.

## Status

Task 9 is completed.

The project evaluates lightweight SAM-family models on the final COGAR-SimRobotics-500 benchmark and compares their accuracy, boundary quality, catastrophic failure rate, and inference speed against SAM ViT-B and FastSAM-S.

The evaluated lightweight SAM variants are:

1. MobileSAM
2. EfficientSAM-Ti

Both models were evaluated zero-shot with box prompts on the full 4,471-object simulated robotic-scene benchmark.

## Visual evidence

![IoU versus FPS trade-off](/outputs/figures/final_report/edge_tradeoff/iou_vs_fps_tradeoff.png)

*Figure: Edge-deployment trade-off for SAM ViT-B, MobileSAM, EfficientSAM-Ti, FastSAM-S, SAM2.1-Tiny, and YOLOv8n-seg supervised reference.*

![Mean IoU model comparison](/outputs/figures/final_report/metrics/mean_iou_by_model_prompt.png)

*Figure: Mean IoU chart showing lightweight SAM variants in context with the broader zero-shot benchmark.*

## Motivation

Full SAM models provide strong segmentation quality but can be too expensive for edge robotic deployment. Lightweight SAM variants aim to preserve promptable segmentation behavior while reducing model size and runtime cost.

This task asks whether lightweight SAM-style models are useful for robotic perception when the deployment hardware is limited.

## Hardware and dataset

| Item | Value |
|---|---|
| Dataset | COGAR-SimRobotics-500 |
| Images | 500 |
| Object instances | 4,471 |
| GPU | NVIDIA GTX 1050 |
| GPU memory | 4 GB VRAM |
| Prompt mode | Box prompts |
| Evaluation type | Zero-shot, no fine-tuning on COGAR-Sim |

## Compared models

| Model | Role |
|---|---|
| SAM ViT-B | Accuracy-oriented SAM reference |
| MobileSAM | Lightweight SAM-style edge candidate |
| EfficientSAM-Ti | Lightweight EfficientSAM-family candidate |
| FastSAM-S | Speed-first segmentation baseline |

## Overall comparison

| Model | Objects | Mean IoU | Median IoU | Boundary F1 | IoU >= 0.90 | IoU >= 0.75 | IoU >= 0.50 | IoU < 0.10 | Mean FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM ViT-B box | 4,471 | 0.9057 | 0.9553 | 0.9356 | 0.7513 | 0.9137 | 0.9703 | 0.0031 | 61.96 |
| MobileSAM box | 4,471 | 0.8656 | 0.9363 | 0.9797 | 0.6285 | 0.8430 | 0.9457 | 0.0045 | 69.52 |
| EfficientSAM-Ti box | 4,471 | 0.8807 | 0.9399 | 0.9109 | 0.6743 | 0.8808 | 0.9571 | 0.0056 | 9.47 |
| FastSAM-S box | 4,471 | 0.6986 | 0.8135 | 0.8920 | 0.2841 | 0.5927 | 0.8085 | 0.0830 | 471.30 |

## Accuracy trade-off

### MobileSAM

MobileSAM gives the strongest lightweight SAM-style edge trade-off in this benchmark.

Compared with SAM ViT-B:

- mean IoU drops from 0.9057 to 0.8656
- median IoU remains high at 0.9363
- catastrophic failure rate remains very low at 0.0045
- FPS is slightly higher than SAM ViT-B in the object-row-level benchmark

Interpretation:

MobileSAM loses some high-precision masks compared with SAM ViT-B, but most object masks remain accurate. It is suitable when the system needs a smaller/lighter SAM-style model and can accept a moderate accuracy reduction.

### EfficientSAM-Ti

EfficientSAM-Ti gives better mean IoU than MobileSAM in this benchmark, but it is slower in the current implementation.

Compared with MobileSAM:

- mean IoU is higher: 0.8807 vs 0.8656
- median IoU is slightly higher: 0.9399 vs 0.9363
- IoU >= 0.75 is higher: 0.8808 vs 0.8430
- FPS is much lower: 9.47 vs 69.52

Interpretation:

EfficientSAM-Ti is a strong lightweight accuracy baseline, but it is not the best speed/edge option on the tested GTX 1050 setup. It may be useful when accuracy is more important than runtime, but MobileSAM is the better practical edge trade-off in this project.

### FastSAM-S comparison

FastSAM-S is much faster than both lightweight SAM variants, but its segmentation quality is substantially lower.

Compared with MobileSAM:

- FastSAM-S FPS is much higher
- mean IoU is much lower: 0.6986 vs 0.8656
- catastrophic failure rate is much higher: 0.0830 vs 0.0045

Interpretation:

FastSAM-S is the speed-first option, but it is less reliable for robotic manipulation where accurate masks are important.

## Category-level observations

Based on the cross-model and EfficientSAM result tables, the hardest categories for lightweight models are:

- cable
- robot_gripper
- tool
- screw

These categories are difficult because they contain thin structures, articulated geometry, small parts, occlusions, and ambiguous boundaries.

The strongest categories are generally:

- plastic_object
- box
- metal_part
- connector

These objects have clearer boundaries and larger visible regions.

## Edge deployment interpretation

### Best lightweight SAM-style trade-off

MobileSAM is the best lightweight SAM-style trade-off for this project.

Reason:

- full 4,471-object evaluation completed
- high median IoU
- low catastrophic failure rate
- faster than SAM ViT-B in the benchmark
- much faster than EfficientSAM-Ti in the current implementation
- suitable when box prompts or object proposals are available

### Best lightweight accuracy option

EfficientSAM-Ti is the best lightweight accuracy-oriented variant among the two lightweight SAM variants tested.

Reason:

- higher mean IoU than MobileSAM
- higher IoU >= 0.75 than MobileSAM
- strong per-category results on most object classes

However, its measured FPS is too low for high-FPS edge deployment on the tested hardware.

### Best speed-first option

FastSAM-S is the best speed-first option.

Reason:

- much higher FPS than all SAM-style variants
- useful when approximate masks are acceptable

However, its lower IoU and higher failure rate make it less suitable for precise robotic manipulation.

## Practical recommendation

For robotic applications on limited hardware:

1. Use YOLOv8n-seg if the object categories are known and supervised training is allowed.
2. Use MobileSAM if a lightweight promptable SAM-style model is needed.
3. Use EfficientSAM-Ti if lightweight promptable accuracy is more important than speed.
4. Use FastSAM-S only when speed is more important than mask quality.
5. Use SAM ViT-B or SAM2.1-Tiny when higher accuracy is required and compute budget allows it.

## Limitations

The reported FPS values for promptable models are object-row-level timing values from the evaluation scripts. They are useful for relative comparison but should not be interpreted as strict end-to-end robot camera-frame FPS.

Only box-prompt evaluation was used for MobileSAM and EfficientSAM-Ti in this task. Additional point-prompt and automatic-mask-generation tests could be added in future work.

The evaluation was performed on a GTX 1050 4 GB laptop GPU, so results may differ on stronger GPUs, embedded GPUs, or optimized inference runtimes.

## Conclusion

Task 9 is completed.

MobileSAM and EfficientSAM-Ti both provide useful lightweight SAM-family baselines for robotic segmentation. MobileSAM provides the best practical edge trade-off because it keeps strong segmentation quality while remaining faster than SAM ViT-B in this benchmark. EfficientSAM-Ti provides stronger mean IoU than MobileSAM but is slower in the current implementation. Therefore, the final recommendation is:

- MobileSAM: best lightweight SAM-style edge trade-off
- EfficientSAM-Ti: best lightweight accuracy-oriented SAM variant
- FastSAM-S: best speed-first but lower-accuracy option
- SAM ViT-B / SAM2.1-Tiny: better accuracy when compute budget allows
