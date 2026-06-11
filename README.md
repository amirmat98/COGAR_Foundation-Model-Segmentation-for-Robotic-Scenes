# COGAR Foundation-Model Segmentation for Robotic Scenes

## Final Assignment Landing Page

- **Project:** Subgroup I2 - Foundation Model Segmentation for Robotic Scenes
- **Assignment:** Zero-Shot Segmentation Benchmark for Robotic Perception (Simulation)
- **Student ID:** 5884715
- **Repository:** `COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes`

This repository contains the final submission version of a simulated robotic-scene
segmentation benchmark. The final benchmark uses COGAR-SimRobotics-500, a
500-image synthetic robotic perception dataset with instance masks, object
categories, challenge labels, prompt points, and bounding boxes.

The final work evaluates zero-shot promptable segmentation models, lightweight
SAM-style variants, and supervised instance-segmentation baselines. The old
debug-subset prototype is no longer the current pipeline.

## Final Status

| Area | Final state |
|---|---|
| Main dataset | COGAR-SimRobotics-500 |
| Images | 500 |
| Object instances | 4,471 |
| Categories | 9 |
| Challenge groups | reflective metal, transparent glass, partial occlusion, small parts, dynamic/manipulation scenes |
| Zero-shot models | SAM ViT-B, SAM2.1-Tiny, FastSAM-S, SAM ViT-H CPU subset |
| Lightweight models | MobileSAM, EfficientSAM-Ti, FastSAM-S |
| Supervised baselines | YOLOv8n-seg, Mask R-CNN ResNet-50 FPN |
| Main metrics | mean IoU, median IoU, boundary F1, per-category IoU, mask AP where applicable, FPS |
| Hardware caveat | SAM ViT-H full CUDA evaluation is infeasible on GTX 1050 4 GB due to CUDA OOM |

## Assignment Checklist

| Task | Status | Evidence |
|---|---|---|
| 1. Simulated robotic scene dataset | Complete | [Task 1](docs/assignment_task1_dataset_completion.md), [dataset summary](docs/final_dataset_summary.md) |
| 2. Simulation environment | Complete | [Task 2](docs/assignment_task2_simulation_environment.md) |
| 3. Robotic platform scope | Documented with limitation | [Task 3](docs/assignment_task3_robotic_platform_scope.md) |
| 4. Zero-shot prompt benchmark | Complete for SAM ViT-B, SAM2.1-Tiny, FastSAM-S; ViT-H hardware-limited | [Task 4](docs/assignment_task4_zero_shot_prompt_benchmark.md) |
| 5. Classical/supervised baselines | Complete | [Task 5](docs/assignment_task5_classical_baselines.md) |
| 6. Evaluation metrics | Complete | [Task 6](docs/assignment_task6_evaluation_metrics.md), [cross-model results](docs/final_cross_model_results.md) |
| 7. Inference speed | Complete | [Task 7](docs/assignment_task7_inference_speed.md) |
| 8. Failure mode analysis | Complete | [Task 8](docs/assignment_task8_failure_mode_analysis.md), [failure analysis](docs/failure_mode_analysis.md) |
| 9. Lightweight SAM edge trade-off | Complete | [Task 9](docs/assignment_task9_lightweight_sam_edge_tradeoff.md) |

## Visual Summary

![Representative COGAR-SimRobotics-500 scenes](/outputs/figures/final_report/dataset/sample_scene_montage.png)

*Figure: Representative simulated robotic scenes used by the final benchmark, including reflective, transparent, small-part, gripper, occlusion, and dynamic-scene cases.*

![Dataset category counts](/outputs/figures/final_report/dataset/category_counts.png)

*Figure: Object-instance distribution across the nine COGAR-SimRobotics-500 categories.*

![Mean IoU comparison](/outputs/figures/final_report/metrics/mean_iou_by_model_prompt.png)

*Figure: Mean IoU comparison across zero-shot models and prompt modes. SAM ViT-H entries are CPU subset reference results only.*

![FPS comparison](/outputs/figures/final_report/speed/fps_comparison.png)

*Figure: Measured FPS comparison on a log scale, showing GPU runs and SAM ViT-H CPU subset timings.*

![IoU and FPS trade-off](/outputs/figures/final_report/edge_tradeoff/iou_vs_fps_tradeoff.png)

*Figure: Accuracy/speed trade-off for deployable model choices. The YOLOv8n-seg point uses mask mAP50-95 rather than IoU and is included as a supervised baseline reference.*

![Failure mode montage](/outputs/figures/final_report/failure_modes/failure_mode_montage.png)

*Figure: Representative failure panels from existing output visualizations. Green is ground truth only, red is prediction only, and yellow is overlap.*

## Dataset Summary

| Property | Value |
|---|---:|
| Dataset | COGAR-SimRobotics-500 |
| Final clean images | 500 |
| Object instances | 4,471 |
| Object categories | 9 |
| Mean objects per image | 8.94 |
| Annotation type | COCO-derived instance masks plus object-level benchmark index |
| Final index | `data/cogar_sim_500_final/annotations/sim_robotic_scenes_index_final_filtered.csv` |

Object categories:

| Category | Instances |
|---|---:|
| robot_gripper | 1,042 |
| plastic_object | 627 |
| metal_part | 555 |
| connector | 531 |
| screw | 427 |
| glass_object | 360 |
| box | 352 |
| tool | 296 |
| cable | 281 |

Challenge groups:

| Challenge group | Instances |
|---|---:|
| small_parts | 1,269 |
| partial_occlusion | 920 |
| dynamic_scene | 797 |
| reflective_metal | 743 |
| transparent_glass | 742 |

The final dataset was generated with the reproducible BlenderProc synthetic
scene pipeline. Isaac Sim was documented as a preferred alternative route, but
was not used for the final 500-image generation on the available GTX 1050
machine or the later Tesla T4 benchmark server. The repo now includes an AWS
Isaac Sim rerun path targeting `g6e.4xlarge` preferred, `g6e.2xlarge` minimum,
or `g6e.8xlarge` if budget/quota allow. Gazebo and Rviz2 were not used because
the synthetic simulation dataset generation satisfied the assignment simulation
requirement. The scenes include a simulated `robot_gripper`; a full Unitree
G1/As2 embodiment is documented as future work due to hardware and scope
limits.

## Model Coverage

| Model | Type | Prompt / mode coverage | Evaluation scope | Status |
|---|---|---|---|---|
| SAM ViT-B | Zero-shot SAM | box, point, auto | 4,471 objects | Complete |
| SAM2.1-Tiny | Zero-shot SAM2 | box, point, auto | 4,471 objects | Complete |
| FastSAM-S | Zero-shot SAM-style | box, point, auto/everything | 4,471 objects | Complete |
| SAM ViT-H | Zero-shot SAM | box, point, auto | CPU subsets only | Hardware-limited |
| MobileSAM | Lightweight SAM | box | 4,471 objects | Complete |
| EfficientSAM-Ti | Lightweight SAM | box | 4,471 objects | Complete |
| YOLOv8n-seg | Supervised instance segmentation | automatic masks | 120/75/75 train/val/test images | Complete |
| Mask R-CNN ResNet-50 FPN | Supervised instance segmentation | automatic masks | 75 test images / 679 objects | Complete |
| DeepLabV3+ | Semantic segmentation | not evaluated | not applicable | Excluded because this benchmark is instance-level |

## Key Zero-Shot Results

Full-dataset box-prompt and lightweight comparison:

| Model / mode | Objects | Mean IoU | Median IoU | Boundary F1 | Mean FPS |
|---|---:|---:|---:|---:|---:|
| SAM2.1-Tiny box | 4,471 | 0.9127 | 0.9553 | 0.9307 | 16.81 |
| SAM ViT-B box | 4,471 | 0.9057 | 0.9553 | 0.9356 | 61.96 |
| EfficientSAM-Ti box | 4,471 | 0.8807 | 0.9399 | 0.9109 | 9.47 |
| MobileSAM box | 4,471 | 0.8656 | 0.9363 | 0.9797 | 69.52 |
| FastSAM-S box | 4,471 | 0.6986 | 0.8135 | 0.8920 | 471.30 |

Prompt-mode comparison for completed full-dataset zero-shot runs:

| Model | Prompt mode | Objects | Mean IoU | Median IoU | Boundary F1 | Mean FPS |
|---|---|---:|---:|---:|---:|---:|
| SAM ViT-B | Box | 4,471 | 0.9057 | 0.9553 | 0.9356 | 61.96 |
| SAM ViT-B | Point | 4,471 | 0.7985 | 0.9125 | 0.8131 | not evaluated |
| SAM ViT-B | Auto | 4,471 | 0.8025 | 0.9422 | 0.8381 | not evaluated |
| SAM2.1-Tiny | Box | 4,471 | 0.9127 | 0.9553 | 0.9307 | 16.81 |
| SAM2.1-Tiny | Point | 4,471 | 0.8658 | 0.9349 | 0.8731 | 16.68 |
| SAM2.1-Tiny | Auto | 4,471 | 0.6403 | 0.8701 | 0.6781 | 2.30 |
| FastSAM-S | Box | 4,471 | 0.6986 | 0.8135 | 0.8920 | 471.30 |
| FastSAM-S | Point | 4,471 | 0.7594 | 0.8883 | 0.7890 | 214.07 |
| FastSAM-S | Auto / Everything | 4,471 | 0.7773 | 0.8914 | 0.8093 | 206.48 |

SAM ViT-H hardware-limited CPU subset:

| Prompt mode | Subset | Device | Mean IoU | Median IoU | Boundary F1 | Mean FPS |
|---|---:|---|---:|---:|---:|---:|
| Box | 25 objects | CPU | 0.9449 | 0.9717 | 0.9637 | 0.1820 |
| Point | 25 objects | CPU | 0.7721 | 0.9547 | 0.7958 | 0.1762 |
| Auto | 42 objects / 5 images | CPU | 0.7302 | 0.9563 | 0.7640 | 0.2118 |

Detailed result pages:

- [SAM ViT-B](docs/final_sam_vit_b_results.md)
- [SAM ViT-H](docs/final_sam_vit_h_results.md)
- [SAM2.1-Tiny](docs/final_sam2_results.md)
- [FastSAM-S](docs/final_fastsam_results.md)
- [MobileSAM](docs/final_mobilesam_results.md)
- [EfficientSAM-Ti](docs/final_efficientsam_results.md)
- [Cross-model comparison](docs/final_cross_model_results.md)

## Supervised Baseline Results

| Model | Evaluation split | Main mask metrics | Speed |
|---|---|---|---|
| YOLOv8n-seg | 75 test images / 679 instances | precision 0.761, recall 0.783, mAP50 0.806, mAP50-95 0.601 | 26.8 ms/image, about 37.3 FPS total; about 90.1 FPS inference-only |
| Mask R-CNN ResNet-50 FPN | 75 test images / 679 objects | mean IoU 0.7462, median IoU 0.8309, boundary F1 0.7218 | 5.5855 image FPS; 50.5675 object FPS |
| DeepLabV3+ | not evaluated | not applicable | Excluded because semantic segmentation does not produce instance masks for this benchmark |

Detailed baseline pages:

- [YOLOv8n-seg](docs/final_yolov8seg_baseline_results.md)
- [Mask R-CNN](docs/final_maskrcnn_baseline_results.md)
- [Task 5 baseline summary](docs/assignment_task5_classical_baselines.md)

## Speed and Real-Time Feasibility

| Model / mode | Speed interpretation |
|---|---|
| FastSAM-S box | Fastest zero-shot SAM-style run, but lower mask quality |
| YOLOv8n-seg | Real-time supervised automatic segmentation on the available GPU |
| MobileSAM box | Best lightweight SAM-style edge trade-off |
| SAM ViT-B box | Strong speed/accuracy balance for prompted segmentation |
| SAM2.1-Tiny box | Highest promptable mean IoU, slower than SAM ViT-B in this implementation |
| EfficientSAM-Ti box | Strong IoU but slower measured FPS than MobileSAM |
| Mask R-CNN | Useful supervised baseline, not real-time on GTX 1050 |
| SAM ViT-H CPU | Valid reference result, not practical for full benchmark runtime |

FPS values for SAM-style object-prompt evaluations are object-row-level timing
values from the benchmark scripts. They are useful for relative comparison, but
they should not be read as strict deployed camera-frame rates.

## Failure Mode Summary

The final failure analysis identifies recurring robotic-scene risks:

| Failure mode | Typical cause |
|---|---|
| Robot grippers and articulated parts | complex shape, holes, contact with objects |
| Cables and thin structures | narrow masks and weak boundaries |
| Screws, connectors, and small parts | small mask area and high boundary sensitivity |
| Transparent glass | weak visible edges and background mixing |
| Reflective metal | specular highlights and unstable appearance |
| Partial occlusion | missing object evidence and overlapping masks |
| Dynamic clutter/manipulation scenes | object contact, changed poses, and local ambiguity |
| Prompt ambiguity | point prompts can select nearby clutter or the wrong instance |
| Automatic-mask proposal errors | prompt-free masks can miss small or occluded target objects |

See [failure mode analysis](docs/failure_mode_analysis.md) and
[Task 8](docs/assignment_task8_failure_mode_analysis.md).

## Recommendation Summary

| Use case | Recommended model |
|---|---|
| Highest promptable full-dataset mean IoU | SAM2.1-Tiny box |
| Best speed/accuracy balance for prompted masks | SAM ViT-B box |
| Best lightweight SAM-style edge trade-off | MobileSAM box |
| Strong lightweight accuracy when speed is less important | EfficientSAM-Ti box |
| Fastest zero-shot approximate masks | FastSAM-S box |
| Supervised automatic deployment baseline | YOLOv8n-seg |
| Classical supervised comparison | Mask R-CNN ResNet-50 FPN |
| Large SAM reference | SAM ViT-H CPU subset only |

For precise robotic manipulation, box prompts remain the most reliable prompt
type when boxes are available. Automatic masks are useful for open-world proposal
generation, but they are less reliable on small parts, grippers, occlusion, and
transparent objects.

See [model recommendation guide](docs/model_recommendation_guide.md).

## Hardware Limitations

The benchmark was completed on constrained local hardware with an NVIDIA GTX
1050 4 GB GPU.

- SAM ViT-H CUDA evaluation failed during image encoding in
  `predictor.set_image(image)` due to CUDA out-of-memory.
- Mixed precision did not resolve the ViT-H CUDA path; FP16 also exposed dtype
  mismatch issues in the local environment.
- SAM ViT-H is therefore reported only as CPU subset evidence for box, point,
  and automatic mask modes.
- Full Unitree G1/As2 embodiment was not generated locally; the dataset instead
  uses simulated robotic manipulation scenes with a visible `robot_gripper`.

These limits are documented as project constraints, not hidden failures.

## Repository Structure

```text
configs/          Reproducible config files and local path examples
data/             Local/generated datasets and lightweight dataset notes
docs/             Assignment task reports, final results, failure analysis
outputs/          Final benchmark evidence and local/generated outputs
scripts/          Thin command-line wrappers for generation, evaluation, analysis
src/cogar_seg/    Reusable dataset, model, prompt, metric, evaluation code
tests/            Lightweight tests that do not require datasets or checkpoints
checkpoints/      Local model weights, ignored by Git
```

Reusable implementation belongs under `src/cogar_seg/`. Scripts should stay as
thin CLI entry points that parse arguments and call package functions.

## Reproducibility Notes

The final submission should be read from the committed docs and lightweight
result summaries. Do not rerun expensive experiments unless a specific result
needs to be regenerated.

Recommended local validation:

```bash
python -m compileall -q src scripts
PYTHONPATH=src pytest -q
```

Dataset generation and normalization entry points are retained for
reproducibility:

```bash
source ~/blenderproc_test/.venv/bin/activate
blenderproc run scripts/blenderproc/generate_cogar_sim_500.py \
  --config configs/blenderproc_dataset.yaml \
  --num-images 500

PYTHONPATH=src python scripts/dataset/normalize_cogar_sim_500.py
```

Object index creation:

```bash
PYTHONPATH=src python scripts/dataset/create_object_index.py \
  --dataset cogar_sim_500 \
  --coco data/cogar_sim_500/annotations/instances_all.json \
  --metadata data/cogar_sim_500/metadata/frame_index.csv \
  --rgb-dir data/cogar_sim_500/rgb \
  --output outputs/indexes/cogar_sim_500_objects.csv
```

## Heavy Asset Policy

The repository should stay lightweight and reproducible.

- Do not commit model weights or checkpoints: `*.pth`, `*.pt`, `*.ckpt`,
  `*.safetensors`.
- Do not commit large raw RGB/depth/mask datasets.
- Keep checkpoints under `checkpoints/` or external storage.
- Keep external model environments and non-repository assets outside this repo.
- Lightweight final tables, JSON summaries, plots, and Markdown reports may be
  committed when they document final benchmark evidence.
- Heavy predicted masks, raw generated images, temporary runs, and checkpoints
  should remain ignored or local.
