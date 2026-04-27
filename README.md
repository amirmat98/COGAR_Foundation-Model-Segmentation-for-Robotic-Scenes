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

