# Slide Deck Outline

Project: **Foundation Model Segmentation for Robotic Scenes**  
Assignment: **Zero-Shot Segmentation Benchmark for Robotic Perception (Simulation)**  
Student id: **5884715**

Recommended repo path:

```text
report/06_slide_deck_outline.md
```

This file defines a practical slide-by-slide structure for the final COGAR presentation. It is designed for a **10–12 minute presentation** and can also be used as the skeleton for the written report.

---

## Presentation Goal

The presentation should not be framed only as a technical comparison of segmentation models. The stronger research framing is:

> This project investigates whether promptable foundation segmentation models can operate as reliable perception modules inside robotic cognitive architectures, especially under robotic-scene challenges such as transparent objects, reflective surfaces, partial occlusion, small parts, dynamic objects, robot-body visibility, and real-time deployment constraints.

The audience should understand five things by the end:

1. Why segmentation is important for robotic cognition and action.
2. Why general image segmentation models may fail in robotic scenes.
3. How the benchmark was formulated scientifically.
4. What the experiments show about quality, robustness, and speed.
5. Which model type should be selected for different robotic scenarios.

---

## Recommended 11-Slide Structure

| Slide | Title | Main role |
|---:|---|---|
| 1 | Title and Research Question | Open the story and define the central question. |
| 2 | Research Problem | Explain why robotic segmentation is difficult. |
| 3 | State of the Art | Situate SAM-family models and classical baselines. |
| 4 | Research Gap | Explain why existing segmentation research is not enough for robotics. |
| 5 | Research Formulation | Present objectives, hypotheses, and methodology. |
| 6 | Dataset and Simulation Design | Show Isaac Unitree G1, BlenderProc, OCID, and challenge categories. |
| 7 | Benchmark Protocol | Explain models, prompts, baselines, and metrics. |
| 8 | Quantitative Results | Present accuracy and speed evidence. |
| 9 | Failure Mode Analysis | Show qualitative failures and explain why they matter. |
| 10 | Cognitive Approach | Connect the project explicitly to COGAR. |
| 11 | Conclusions and Recommendations | Match conclusions to evidence and give model-selection guidance. |

---

# Slide 1 — Title and Research Question

## Slide title

**Zero-Shot Foundation Model Segmentation for Robotic Scene Understanding**

## Slide content

- Assignment 2: Zero-Shot Segmentation Benchmark for Robotic Perception
- Student id: 5884715
- Main question:

> Can promptable foundation segmentation models provide reliable zero-shot object masks for robotic perception in challenging simulated scenes?

## Visual suggestion

Use one strong image or montage from the benchmark:

```text
outputs/final_benchmark_assets/plots/dataset_examples.png
```

If the exact figure is unavailable, use a simple three-column visual:

```text
Isaac Unitree G1 scene | BlenderProc scene | OCID clutter scene
```

## Speaker notes

Start with the robotic motivation, not the model names. A robot needs to know where objects are before it can grasp, avoid, track, inspect, or reason about them. Segmentation masks are therefore not only image outputs; they are possible inputs to robot decision-making.

---

# Slide 2 — Research Problem

## Slide title

**Research Problem: Segmentation Is Harder in Robotic Scenes**

## Slide content

- Robots need object-level perception for manipulation, navigation, tracking, and interaction.
- Robotic scenes contain visual conditions that are difficult for segmentation:
  - reflective metal,
  - transparent glass,
  - partial occlusion,
  - small screws/connectors,
  - moving objects,
  - robot-body visibility,
  - cluttered backgrounds.
- The problem is to test whether zero-shot foundation models remain reliable under these conditions.

## Visual suggestion

Use a compact grid of challenge examples:

```text
Transparent object | Reflective object | Occluded object | Small part | Dynamic object
```

## Speaker notes

Make clear that the benchmark is not about normal internet images. The problem is embodied robotic perception: objects are seen from robot-centered viewpoints, under clutter, occlusion, material effects, and real-time constraints.

---

# Slide 3 — State of the Art

## Slide title

**State of the Art: From Classical Segmentation to Promptable Foundation Models**

## Slide content

- **SAM**: promptable segmentation model for zero-shot transfer.
- **SAM2**: extends promptable segmentation to images and video with memory.
- **FastSAM**: faster segment-anything-style approach.
- **MobileSAM / EfficientSAM**: lightweight SAM-style models for constrained deployment.
- **Mask R-CNN, DeepLabV3+, YOLOv8-seg**: supervised baselines for instance/semantic segmentation.

## Visual suggestion

Use a model-family diagram:

```text
Classical supervised models
        ↓
Promptable foundation models
        ↓
Lightweight / edge-oriented variants
```

## Speaker notes

Do not spend too much time explaining each architecture. The purpose of this slide is to show the field has moved from task-specific supervised segmentation to general promptable segmentation, but robotics still requires validation under embodied conditions.

---

# Slide 4 — Research Gap

## Slide title

**Research Gap: General Segmentation Is Not Automatically Robotic Perception**

## Slide content

- Foundation models are trained for broad visual generalization.
- Robotic perception requires situated, action-relevant, and time-constrained perception.
- Open questions:
  - Do zero-shot masks remain accurate on robotic materials and clutter?
  - Which prompt mode is most useful for robot tasks?
  - Are foundation models fast enough for closed-loop use?
  - Do lightweight variants preserve enough quality for edge deployment?

## Visual suggestion

Use a contrast diagram:

```text
Generic image segmentation
  image → mask

Robotic cognitive perception
  sensor input → attention → mask → object representation → action decision
```

## Speaker notes

This is the transition from state of the art to your contribution. The gap is not that SAM does not exist. The gap is that we need to know when a segmentation foundation model is trustworthy as a component of robotic cognition.

---

# Slide 5 — Research Formulation

## Slide title

**Research Formulation**

## Slide content

**Objective**

Benchmark zero-shot foundation segmentation models for robotic scene understanding.

**Hypotheses**

1. Box prompts will produce stronger masks than point prompts and automatic masks.
2. SAM/SAM2 will provide high mask quality but may be computationally expensive.
3. Lightweight SAM variants improve deployability but may reduce robustness.
4. Supervised baselines can be competitive when small labeled subsets are available.
5. Transparent, reflective, occluded, small, and dynamic objects will cause frequent failures.

## Visual suggestion

Use a simple experimental pipeline:

```text
Simulated/real robotic datasets
        ↓
Zero-shot SAM-family models + supervised baselines
        ↓
Metrics: mIoU, Boundary F1, Mask AP, per-category IoU, FPS
        ↓
Failure analysis and recommendations
```

## Speaker notes

Frame the project as an empirical research study. The experiments are designed to test accuracy, robustness, and deployment feasibility, not only to produce a leaderboard.

---

# Slide 6 — Dataset and Simulation Design

## Slide title

**Dataset and Simulation Design**

## Slide content

| Dataset | Type | Images | Role |
|---|---:|---:|---|
| Isaac official Unitree G1 | Synthetic, Isaac Sim | 1000 | Main robot-centered simulation benchmark |
| BlenderProc COGAR-SimRobotics-1000 | Synthetic, BlenderProc | 1000 | Secondary synthetic benchmark |
| OCID | Real RGB-D clutter dataset | 2390 | Real-world robustness/domain-gap reference |

Challenge groups:

- reflective objects,
- transparent objects,
- partial occlusion,
- small parts,
- dynamic objects,
- cluttered robot scenes.

## Visual suggestion

Use dataset examples and a small challenge-category legend:

```text
outputs/final_benchmark_assets/plots/dataset_examples.png
```

## Speaker notes

Emphasize why simulation was used: it allows controlled generation of rare or difficult robotic conditions, while still producing dense annotations for objective evaluation. OCID is useful as a real-world robustness reference.

---

# Slide 7 — Benchmark Protocol

## Slide title

**Benchmark Protocol**

## Slide content

**Zero-shot models**

- SAM ViT-H
- SAM ViT-B
- SAM2
- FastSAM
- MobileSAM
- EfficientSAM

**Prompt modes**

- point prompt,
- box prompt,
- automatic mask generation.

**Supervised baselines**

- YOLOv8-seg,
- Mask R-CNN,
- DeepLabV3+.

**Metrics**

- mIoU,
- boundary F1,
- mask AP / AP50 / AP75,
- per-category IoU,
- GPU/CPU FPS.

## Important note for the slide

Point and box prompts are **oracle-prompt evaluations**. They test mask quality when a target cue is available. In a real robot, that cue must come from an upstream detector, tracker, grasp planner, human instruction, or task prior.

## Visual suggestion

Use a three-panel prompt-mode diagram:

```text
Point prompt → target cue from one pixel
Box prompt → target cue from bounding box
Automatic → model proposes masks without a prompt
```

## Speaker notes

This slide prevents overclaiming. Box-prompted SAM is not the same as fully autonomous perception. It is a useful setting because robots often have task cues or upstream modules, but it must be explained honestly.

---

# Slide 8 — Quantitative Results

## Slide title

**Quantitative Results: Accuracy, Robustness, and Speed**

## Slide content

Use the final corrected common-test results when available:

```text
outputs/task6_evaluation/zero_shot/test/summary.csv
outputs/task6_evaluation/baselines/test/summary.csv
outputs/task7_inference_speed/summary.csv
outputs/task9_lightweight_sam/summary/summary.json
```

Recommended plots:

```text
outputs/final_benchmark_assets/plots/zero_shot_miou_heatmap.png
outputs/final_benchmark_assets/plots/baseline_miou_bars.png
outputs/final_benchmark_assets/plots/cuda_speed_quality_scatter.png
outputs/final_benchmark_assets/plots/lightweight_sam_tradeoff_cuda.png
```

Key result messages to use only if supported by the final corrected outputs:

- Box prompts usually give the strongest segmentation quality.
- SAM/SAM2 are strong when mask quality is the priority.
- Supervised baselines may be preferable for real-time GPU throughput.
- Lightweight SAM variants can reduce deployment cost but require quality-speed trade-off analysis.

## Speaker notes

Do not overload the audience with every metric. Show one accuracy plot and one speed-quality plot. Explain the interpretation: high mIoU alone is not enough for robotics if FPS is too low, and high FPS is not enough if small or transparent objects are missed.

## Evidence discipline

If the final corrected common-test outputs are not yet regenerated, label results as:

```text
Archived / preliminary benchmark outputs; final common-test comparison pending.
```

Do not call legacy validation/full-dataset metrics final comparative evidence.

---

# Slide 9 — Failure Mode Analysis

## Slide title

**Failure Mode Analysis: Where Models Break**

## Slide content

Main failure categories:

- transparent objects: weak or missing boundaries,
- reflective metal: confusing highlights and object edges,
- small screws/connectors: under-segmentation or missed objects,
- partial occlusion: incomplete masks,
- dynamic/robot-body scenes: false positives or merged masks,
- clutter: over-segmentation and object merging.

## Visual suggestion

Use qualitative examples from:

```text
outputs/task8_failure_analysis/
```

A strong layout:

```text
RGB image | Ground truth | Prediction | Error explanation
```

## Speaker notes

This slide is important because it connects quantitative metrics to robotic consequences. A small segmentation error may be critical if it affects grasping, collision avoidance, or object tracking.

---

# Slide 10 — Cognitive Approach

## Slide title

**Cognitive Approach: Segmentation as a Robotic Perception Module**

## Slide content

Segmentation is treated as a cognitive interface between visual input and action.

| COGAR concept | Project connection |
|---|---|
| Embodiment | Evaluation uses robot-centered scenes and Unitree G1 simulation. |
| Attention | Point and box prompts simulate task-driven visual attention. |
| Scene representation | Masks can become object-level regions for reasoning. |
| Perception-action loop | Mask quality affects grasping, tracking, and navigation. |
| Real-time cognition | FPS and latency determine whether perception can run inside control loops. |
| Robustness | Failure analysis identifies unsafe or unreliable perception conditions. |

## Visual suggestion

Use a cognitive-perception pipeline:

```text
Camera/RGB-D input
        ↓
Foundation segmentation model
        ↓
Object masks / regions
        ↓
Attention + scene representation
        ↓
Action selection / manipulation / navigation
```

## Speaker notes

This is the slide that makes the project clearly belong to COGAR. The point is not only whether SAM produces good masks. The point is whether the mask can be trusted as part of an embodied cognitive architecture.

---

# Slide 11 — Conclusions and Recommendations

## Slide title

**Conclusions and Recommendations**

## Slide content

Evidence-based conclusions:

- Foundation segmentation models can support robotic perception, especially when prompts are available.
- Box prompting is the most useful setting for target-specific robotic tasks, but it assumes an upstream cue.
- Automatic mask generation is more autonomous, but usually harder in cluttered robotic scenes.
- Real-time deployment requires speed analysis, not only accuracy analysis.
- Lightweight models are promising for edge deployment, but must be selected according to the acceptable quality loss.
- Transparent, reflective, occluded, small, and dynamic objects remain important failure cases.

Recommended model-selection logic:

| Robotic scenario | Recommended direction |
|---|---|
| Highest mask quality, offline or slow loop | SAM / SAM2 with prompts |
| Prompt-guided manipulation | Box-prompted SAM-family model |
| Real-time GPU loop | YOLOv8-seg or DeepLabV3+ if trained data is available |
| Edge or constrained deployment | MobileSAM / EfficientSAM after speed-quality validation |
| Fully autonomous object discovery | Automatic masks, but with failure checks |

## Final sentence

> Foundation segmentation models are useful perception modules for robotic cognitive architectures, but they do not remove the need for prompt generation, domain-specific validation, speed analysis, and failure-mode awareness.

## Speaker notes

End with a balanced conclusion. Do not say that foundation models solve robotic perception. Say that they are powerful modules, but robotics requires situated validation and careful integration.

---

## Timing Plan

For a 10–12 minute talk:

| Slide | Approximate time |
|---:|---:|
| 1 | 45 seconds |
| 2 | 1 minute |
| 3 | 1 minute |
| 4 | 1 minute |
| 5 | 1 minute |
| 6 | 1 minute |
| 7 | 1 minute |
| 8 | 1.5–2 minutes |
| 9 | 1 minute |
| 10 | 1 minute |
| 11 | 1 minute |

If the presentation must be shorter, merge:

- Slide 3 and Slide 4,
- Slide 6 and Slide 7,
- Slide 10 and Slide 11.

---

## Minimal 7-Slide Version

If time is limited, use this compressed structure:

| Slide | Title |
|---:|---|
| 1 | Title and Research Question |
| 2 | Research Problem and State of the Art |
| 3 | Research Formulation |
| 4 | Dataset and Benchmark Protocol |
| 5 | Quantitative Results |
| 6 | Failure Modes and Cognitive Approach |
| 7 | Conclusions and Recommendations |

---

## What to Avoid

Avoid these weak or risky statements:

```text
SAM solves robotic perception.
```

Better:

```text
SAM-family models can support robotic perception when prompts, validation, and deployment constraints are handled carefully.
```

Avoid:

```text
Box-prompted performance means the model is fully autonomous.
```

Better:

```text
Box-prompted performance measures mask quality when a target cue is available from an upstream robotic module or oracle evaluation.
```

Avoid:

```text
The fastest model is the best model.
```

Better:

```text
The best model depends on the robotic use case: quality, real-time control, edge deployment, or autonomous object discovery.
```

---

## Files to Use for Figures

Preferred figure sources from the repo:

```text
outputs/final_benchmark_assets/plots/dataset_examples.png
outputs/final_benchmark_assets/plots/zero_shot_miou_heatmap.png
outputs/final_benchmark_assets/plots/baseline_miou_bars.png
outputs/final_benchmark_assets/plots/cuda_speed_quality_scatter.png
outputs/final_benchmark_assets/plots/lightweight_sam_tradeoff_cuda.png
outputs/final_benchmark_assets/plots/challenge_group_weighted_iou.png
outputs/final_benchmark_assets/plots/zero_shot_dataset_prompt_winners.png
outputs/task8_failure_analysis/
```

Use only figures generated from the final corrected common-test outputs for final comparative claims.

---

## Reference Anchors

Use these sources in slide speaker notes and the final report:

- Segment Anything: https://arxiv.org/abs/2304.02643
- SAM2: https://arxiv.org/abs/2408.00714
- FastSAM: https://arxiv.org/abs/2306.12156
- MobileSAM: https://github.com/ChaoningZhang/MobileSAM
- EfficientSAM: https://arxiv.org/abs/2312.00863
- Mask R-CNN: https://arxiv.org/abs/1703.06870
- DeepLabV3+: https://arxiv.org/abs/1802.02611
- Current project report: https://github.com/amirmat98/COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes/blob/main/REPORT.md
- Task 4 zero-shot protocol: https://github.com/amirmat98/COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes/blob/main/docs/tasks/task4_zero_shot_sam.md
- Task 6 evaluation protocol: https://github.com/amirmat98/COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes/blob/main/docs/tasks/task6_evaluation.md

---

## Next Development Step

After this outline, the next file should be the report skeleton:

```text
report/final_report.md
```

That file should combine the previous section files into one coherent written report.
