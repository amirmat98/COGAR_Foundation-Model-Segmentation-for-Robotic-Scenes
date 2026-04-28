# COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes

## Subgroup I2: Foundation Model Segmentation for Robotic Scenes

## Assignment 2: Zero‑Shot Segmentation Benchmark for Robotic Perception (SIMULATION)
### Student id: 5884715

What to do: Conduct a systematic benchmark of SAM, SAM2, FastSAM, MobileSAM, and EfficientSAM as zero-shot segmentation models for robotic scene understanding, evaluating their robustness against challenges specific to robotics such as reflective surfaces, transparent objects, partial occlusions, small parts, and dynamic scenes. 
1) Create or curate a robotic scene dataset in simulation with diverse challenges, including reflective metal, transparent glass, partial occlusions, small screws/connectors, and moving objects (about 500 annotated images).
2) Use simulation environments such as Isaac Sim (better), Gazebo, and/or Rviz2 to generate and organize the robotic scenes.
3) Use simulated robotic platforms such as Unitree As2 EDU or Unitree G1 EDU to produce meaningful benchmark scenes for segmentation.
4) Run SAM (ViT-H, ViT-B), SAM2, and FastSAM in zero-shot mode on the dataset using point prompts, box prompts, and automatic mask generation.
5) Run classical baseline models such as Mask R-CNN, DeepLabV3+, and YOLOv8-seg fine-tuned on a small subset for comparison.
6) Evaluate the models using standard metrics including mIoU, boundary F1, mask AP, and per-category IoU.
7) Measure inference speed (FPS) on GPU and CPU to assess real-time feasibility for robotic applications.
8) Analyze failure modes qualitatively, identifying where and why the segmentation models fail in robotic scenarios.
9) Test whether lightweight SAM variants such as MobileSAM and EfficientSAM can provide a good trade-off for edge deployment.

Software needed: PyTorch, SAM / SAM2 / FastSAM / MobileSAM / EfficientSAM, Detectron2, Ultralytics YOLOv8, OpenCV, COCO evaluation tools, Isaac Sim, Gazebo, Rviz2, Python

Research needed: SAM architecture and prompting strategies, zero-shot segmentation literature, robotic perception benchmarks, synthetic data generation in simulation, domain gap analysis, model distillation for edge AI

Deliverables: Simulated annotated robotic scene dataset, full benchmark results with tables and plots, failure mode analysis report, recommendation guide for which segmentation model to use in different robotic scenarios


<hr style="height:4px; border:none; background-color:white;">

This repository is part of the COGAR project on zero-shot segmentation for robotic perception.  
The original assignment targets benchmarking SAM, SAM2, FastSAM, MobileSAM, and EfficientSAM on robotic scenes with challenges such as occlusion, clutter, reflective/transparent objects, and small parts.

## Work Completed So Far

Instead of starting directly from simulation, I first built a reproducible benchmark pipeline using the **OCID Object Clutter Indoor Dataset**, which provides cluttered RGB-D robotic scenes with object annotations.

### Block 1 — Dataset Setup
- Selected OCID as the first dataset for debugging and benchmarking.
- Configured dataset paths through `configs/paths.yaml`.
- Used the debug sequence: `YCB10/table/top/mixed/seq21`.
- Created an image-level RGB/label index with 11 image-label pairs.

### Block 2 — Object-Level Ground Truth
- Extracted object instances from OCID label images.
- Created an object-level index with 77 initial object instances.
- Filtered the index to 52 usable object instances.
- Exported one binary ground-truth mask per object.
- Final CSV: `outputs/indexes/ocid_debug_seq21_objects_filtered_with_masks.csv`.

### Block 3 — Single-Object SAM Inference
- Implemented `scripts/run_sam_box_prompt.py`.
- Ran SAM ViT-B on selected OCID objects using bounding-box prompts.
- Added CUDA/GPU support for the local GTX 1050 setup.
- Example result for row 0: SAM score = 0.9687, IoU = 0.8818.

### Block 4 — Batch SAM Evaluation
- Implemented `scripts/run_sam_box_prompt_batch.py`.
- Ran SAM ViT-B with box prompts on all 52 filtered OCID objects.
- Saved predicted masks, visualizations, and a result CSV.
- Final results: mean IoU = 0.8495, median IoU = 0.8784, min IoU = 0.7087, max IoU = 0.9126, mean SAM score = 0.9629.

## Current Status

The project now has a working OCID-to-SAM evaluation pipeline: dataset indexing, object-level mask extraction, SAM box-prompt inference, CUDA execution, mask visualization, and IoU-based evaluation.

Next step: analyze failure cases and extend the benchmark to more prompts, more sequences, and additional models such as SAM2, FastSAM, MobileSAM, and EfficientSAM.