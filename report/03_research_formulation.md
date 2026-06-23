# Research Formulation

Recommended repo path:

```text
presentation/03_research_formulation.md
```

This file defines the research formulation for the final presentation and report. It turns the project from a technical benchmark into a research study with a clear question, objectives, hypotheses, methodology, variables, evaluation logic, and limitations.

---

## 1. Slide Version

### Research Question

> Can promptable foundation segmentation models provide reliable zero-shot object masks for robotic perception in challenging simulated scenes, and how do they compare with lightweight and supervised segmentation models in terms of accuracy, robustness, and real-time feasibility?

### Objective

Evaluate whether SAM-family foundation models can act as segmentation modules for robotic scene understanding under realistic robotic challenges such as transparent objects, reflective surfaces, partial occlusion, small parts, robot-body visibility, dynamic scenes, and deployment constraints.

### Hypotheses

1. **Prompted segmentation will outperform automatic mask generation** because point and box prompts provide task-relevant attention cues.
2. **Box prompts will be more reliable than point prompts** because bounding boxes provide stronger spatial constraints.
3. **SAM/SAM2 will provide stronger mask quality than lightweight variants**, especially on complex boundaries and cluttered scenes.
4. **Lightweight variants will offer better deployment trade-offs**, but may lose robustness or boundary quality.
5. **Supervised baselines may be more suitable for real-time robotic control** when a small labeled subset is available.
6. **Transparent, reflective, occluded, small, and dynamic objects will remain the dominant failure cases** for zero-shot segmentation models.

### Methodology Summary

1. Generate and curate robotic-scene datasets in simulation and real-world clutter.
2. Run zero-shot SAM-family models using point, box, and automatic prompt modes.
3. Train classical segmentation baselines on small labeled subsets.
4. Evaluate quality using mIoU, boundary F1, mask AP, and per-category IoU.
5. Evaluate deployment feasibility using GPU and CPU FPS.
6. Analyze qualitative failure modes and connect them to robotic perception risks.
7. Produce model-selection recommendations for different robotic use cases.

---

## 2. Report Version

### 2.1 Research Question

The central research question of this project is:

> Can promptable foundation segmentation models provide reliable zero-shot object masks for robotic perception in challenging simulated scenes, and how do they compare with lightweight and supervised segmentation models in terms of accuracy, robustness, and real-time feasibility?

This question is motivated by the growing use of foundation models in visual perception and by the specific demands of robotic systems. In robotics, segmentation is not only a visual output; it can become an input to grasping, tracking, navigation, manipulation, object memory, and task-level decision making. Therefore, the value of a segmentation model depends not only on average accuracy, but also on whether it remains reliable under conditions that occur in embodied interaction with the physical world.

The project investigates this question using a benchmark built around robotic-scene challenges: transparent glass, reflective metal, partial occlusion, small screws and connectors, moving objects, cluttered scenes, and robot-centered viewpoints. These conditions are important because they directly affect the quality of object-level scene representations available to a cognitive robotic architecture.

---

### 2.2 Research Objectives

The project has six main objectives.

#### Objective 1: Build or curate a robotic-scene benchmark

The first objective is to create or organize datasets that represent the visual difficulty of robotic perception. The final benchmark uses:

- Isaac official Unitree G1 scenes
- BlenderProc COGAR-SimRobotics scenes
- OCID real clutter scenes

The synthetic datasets allow controlled evaluation of specific robotic challenges, while OCID provides a real-world cluttered-object benchmark for domain-gap analysis.

#### Objective 2: Evaluate zero-shot foundation segmentation models

The second objective is to test whether promptable segmentation models can generalize to robotic scenes without dataset-specific fine-tuning. The benchmark includes:

- SAM ViT-H
- SAM ViT-B
- SAM2
- FastSAM

These models are evaluated in zero-shot mode because the research question focuses on generalization and promptability rather than full retraining.

#### Objective 3: Evaluate prompt strategies

The third objective is to compare different prompting strategies:

- point prompts
- box prompts
- automatic mask generation

This is important because prompts can be interpreted as a form of task-driven attention. In a robot, such attention may come from a detector, tracker, grasp planner, human operator, or task prior.

#### Objective 4: Compare foundation models with supervised baselines

The fourth objective is to compare zero-shot foundation models against classical segmentation models trained on a small labeled subset:

- YOLOv8-seg
- Mask R-CNN
- DeepLabV3+

This comparison tests whether a small amount of supervised adaptation can outperform zero-shot generalization, especially for real-time robotic use.

#### Objective 5: Evaluate deployment feasibility

The fifth objective is to measure whether each model is practical for robotic deployment. For this reason, the benchmark includes GPU and CPU inference-speed measurements, including FPS and latency. This matters because robotic perception often runs inside a perception-action loop where delayed perception can reduce the quality of action selection and control.

#### Objective 6: Analyze failure modes

The sixth objective is to identify where and why the models fail. The failure-mode analysis focuses on:

- transparent objects
- reflective surfaces
- small or thin parts
- partial occlusion
- robot-body confusion
- cluttered backgrounds
- dynamic or moving objects

This analysis is necessary because average metrics alone do not show whether a model is safe or reliable for downstream robotic reasoning.

---

## 3. Hypotheses

### Hypothesis 1: Prompted modes will outperform automatic mask generation

Point and box prompts provide additional spatial information about the object of interest. Therefore, prompted segmentation is expected to produce more accurate masks than automatic mask generation, especially in cluttered scenes.

**Expected result:** point and box modes should generally outperform automatic mode in mIoU and mask AP.

**Robotic interpretation:** prompt modes correspond to task-driven attention. A robot that already has a target cue can use the segmentation model more effectively than a robot that must discover all objects without guidance.

---

### Hypothesis 2: Box prompts will be more reliable than point prompts

A point prompt gives only one positive cue inside the object, while a box prompt gives approximate object extent. Therefore, box prompts are expected to produce more stable masks, especially for larger objects, cluttered backgrounds, and objects with ambiguous texture.

**Expected result:** box-prompt segmentation should generally outperform point-prompt segmentation.

**Robotic interpretation:** if an upstream detector, tracker, or planner can provide a bounding box, a foundation segmentation model can become a high-quality mask refinement module.

---

### Hypothesis 3: SAM/SAM2 will provide stronger quality than lightweight variants

SAM and SAM2 use larger foundation-model architectures and are expected to produce stronger masks on difficult scenes. Lightweight variants are expected to reduce computational cost, but may lose performance on complex boundaries, small parts, transparent objects, and occlusions.

**Expected result:** SAM/SAM2 should usually rank higher in segmentation quality, while MobileSAM and EfficientSAM may rank better in speed-size trade-off.

**Robotic interpretation:** high-capacity models are useful for offline mapping, annotation, or high-quality manipulation planning, while lightweight models may be more appropriate for edge deployment.

---

### Hypothesis 4: Supervised baselines may be more suitable for real-time closed-loop robotics

Classical supervised models trained on a small labeled subset may adapt more directly to the target dataset and may run faster than large foundation models.

**Expected result:** YOLOv8-seg and DeepLabV3+ may be more practical for real-time robotic applications when a small amount of labeled data is available.

**Robotic interpretation:** zero-shot flexibility is valuable, but a closed-loop robot may prefer a smaller, faster, task-specific model when the operating domain is known.

---

### Hypothesis 5: Robotic challenge groups will reveal failure modes hidden by aggregate metrics

Transparent, reflective, occluded, small, and dynamic objects are expected to cause more errors than standard opaque and isolated objects.

**Expected result:** per-category IoU, challenge-group performance, and qualitative overlays should reveal weaknesses that are not fully visible from overall mIoU.

**Robotic interpretation:** a model can look strong on average but still be unreliable for the specific object categories that matter for manipulation.

---

## 4. Variables and Evaluation Design

### Independent Variables

The benchmark varies the following factors:

| Factor | Values |
|---|---|
| Dataset | Isaac Unitree G1, BlenderProc COGAR-SimRobotics, OCID |
| Model family | SAM, SAM2, FastSAM, MobileSAM, EfficientSAM, supervised baselines |
| Prompt mode | point, box, automatic |
| Deployment device | GPU, CPU |
| Challenge group | reflective, transparent, occluded, small parts, dynamic objects, clutter |

### Dependent Variables

The measured outcomes are:

| Outcome | Meaning |
|---|---|
| mIoU | Average overlap between predicted and ground-truth masks |
| Boundary F1 | Quality of predicted object boundaries |
| Mask AP / AP50 / AP75 | Instance segmentation quality under COCO-style evaluation |
| Per-category IoU | Performance by object category |
| FPS / latency | Real-time feasibility |
| Failure cases | Qualitative robustness and safety evidence |

### Controlled Conditions

To keep comparisons fair, the benchmark uses a common evaluation layer and the same held-out test split for final comparisons. The training and validation subsets are used only for supervised baseline training and checkpoint selection. Final quality comparisons should be made on the held-out test split.

---

## 5. Methodology

### Step 1: Dataset generation and curation

The project uses simulated and real robotic-scene data. The Isaac dataset provides robot-centered scenes using the Unitree G1 platform. BlenderProc provides additional controlled synthetic scenes. OCID provides real cluttered scenes for testing robustness and domain gap.

The datasets are converted or stored in COCO-style instance annotation format where needed, allowing the same evaluation tools to be used across multiple datasets and model families.

---

### Step 2: Prompt generation

For point-prompt and box-prompt evaluation, prompts are generated from the ground-truth annotations.

- A point prompt is generated from the foreground mask, near the object centroid.
- A box prompt is generated from the COCO bounding box.
- Automatic mode does not receive an object-specific prompt.

This design isolates segmentation quality under known-target conditions. However, point and box prompt results should be described as oracle-prompt evaluations. In a real robotic system, these prompts would need to come from an upstream perception or planning module.

---

### Step 3: Zero-shot model inference

The SAM-family models are run without dataset-specific training. The goal is to test generalization to robotic scenes rather than to optimize each model for one dataset.

The zero-shot models are evaluated across datasets and prompt modes to measure how much prompt type, dataset domain, and visual challenge affect performance.

---

### Step 4: Supervised baseline training

YOLOv8-seg, Mask R-CNN, and DeepLabV3+ are trained using small labeled subsets. These baselines provide a practical comparison against zero-shot models.

This answers an important robotics question: when is it better to use a large general-purpose model, and when is it better to train a smaller model on a limited amount of task-specific data?

---

### Step 5: Quality evaluation

Segmentation quality is evaluated with:

- mIoU
- boundary F1
- mask AP / AP50 / AP75
- per-category IoU

These metrics measure both region overlap and boundary quality. This is important because robotic manipulation often depends on accurate object boundaries, not only rough localization.

---

### Step 6: Speed evaluation

The benchmark measures inference speed on GPU and CPU. The purpose is to evaluate whether each model is feasible for real-time robotic systems.

Speed is interpreted together with mask quality. A model with the best mIoU may not be the best choice for closed-loop control if it is too slow.

---

### Step 7: Failure-mode analysis

Quantitative metrics are complemented by qualitative failure analysis. Representative failure cases are grouped by robotic challenge type. This helps explain why certain models fail and how those failures would affect downstream robot behavior.

---

## 6. Connection to Cognitive Architectures for Robotics

This research formulation connects to COGAR because the benchmark treats segmentation as part of a cognitive robotic architecture.

In cognitive robotics, perception is not passive image analysis. Perception supports attention, memory, planning, action selection, and interaction with the environment. In this project, segmentation masks are interpreted as object-level perceptual representations that may feed downstream modules.

### Cognitive interpretation of the benchmark

| Research element | COGAR interpretation |
|---|---|
| Point and box prompts | Task-driven attention |
| Object masks | Object-level scene representation |
| Unitree G1 simulation | Embodied perception |
| Transparent/reflective/occluded objects | Situated perceptual uncertainty |
| FPS and latency | Real-time perception-action loop |
| Failure analysis | Limits of reliable action selection |
| Lightweight variants | Edge cognition and resource-bounded robotics |

The cognitive approach is therefore not to treat SAM as a standalone computer vision model, but to evaluate whether foundation segmentation can become a reliable perception component in a robot's cognitive architecture.

---

## 7. Congruence Rules for Results and Conclusions

The presentation and report should keep the conclusions aligned with the available evidence.

### Safe conclusions

These conclusions are appropriate if supported by final test results:

- Box-prompted foundation models are effective when a target cue is available.
- Automatic mask generation is less reliable for autonomous object discovery in cluttered robotic scenes.
- Lightweight SAM variants can be useful for edge deployment, but speed-quality trade-offs must be measured.
- Supervised baselines remain important when real-time performance and domain-specific adaptation are required.
- Transparent, reflective, occluded, and small objects remain important failure cases.

### Conclusions to avoid

Avoid claims such as:

- Foundation models solve robotic perception.
- SAM is fully autonomous object understanding.
- High mIoU alone proves a model is suitable for robotic deployment.
- Box-prompt results are equivalent to fully autonomous perception.
- Lightweight models are always better for robotics.

### Correct wording

Use this:

> The benchmark suggests that foundation segmentation models can support robotic perception when prompt generation, robustness validation, speed constraints, and failure modes are explicitly considered.

Avoid this:

> Foundation models replace the perception system of a robot.

---

## 8. Suggested Slide Content

### Slide title

Research Formulation

### Slide bullets

- **Question:** Can promptable foundation segmentation models provide reliable zero-shot masks for robotic scenes?
- **Objective:** Compare accuracy, robustness, and speed across foundation models, lightweight variants, and supervised baselines.
- **Hypotheses:** prompts improve quality; box prompts are strongest; lightweight models trade accuracy for deployability.
- **Method:** simulate/categorize robotic scenes, run zero-shot inference, train small supervised baselines, evaluate quality + speed + failures.
- **COGAR link:** segmentation is evaluated as a perception module inside a robot's perception-action loop.

### Speaker notes

The research is formulated as a benchmark because the goal is not to propose a new segmentation model, but to evaluate whether existing foundation models are suitable for robotic perception. The important point is that robotic perception has different requirements from ordinary image segmentation. A robot needs masks that are accurate enough for downstream actions, fast enough for control, and robust enough under difficult physical conditions. Therefore, the benchmark combines segmentation metrics, speed measurements, and failure analysis.

---

## 9. Report-Ready Paragraph

The research is formulated as a systematic benchmark of promptable foundation segmentation models for robotic scene understanding. The central question is whether models such as SAM, SAM2, FastSAM, MobileSAM, and EfficientSAM can provide reliable zero-shot masks in challenging robotic scenes, and how they compare with supervised baselines trained on small labeled subsets. The study evaluates performance across simulated and real cluttered datasets, prompt modes, model families, challenge groups, and hardware settings. The formulation is intentionally multi-dimensional: segmentation quality is measured with mIoU, boundary F1, mask AP, and per-category IoU; deployment feasibility is measured with GPU and CPU inference speed; and reliability is studied through qualitative failure analysis. This structure allows the conclusions to remain congruent with the evidence: foundation models are evaluated not only by average mask quality, but also by whether they can support embodied robotic perception under real-time and robustness constraints.

---

## 10. Key Takeaway

The project is not only asking:

> Which segmentation model has the best mIoU?

It is asking:

> Which segmentation model is appropriate for which robotic perception role, under which prompt assumption, challenge condition, and deployment constraint?

That is the correct research formulation for a COGAR-oriented benchmark.

---

## 11. References

- Project report: `REPORT.md`
  - https://github.com/amirmat98/COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes/blob/main/REPORT.md
- Task 4 zero-shot SAM-family segmentation:
  - https://github.com/amirmat98/COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes/blob/main/docs/tasks/task4_zero_shot_sam.md
- Task 6 evaluation protocol:
  - https://github.com/amirmat98/COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes/blob/main/docs/tasks/task6_evaluation.md
- Segment Anything:
  - https://arxiv.org/abs/2304.02643
- SAM2:
  - https://arxiv.org/abs/2408.00714
- FastSAM:
  - https://arxiv.org/abs/2306.12156
- EfficientSAM:
  - https://arxiv.org/abs/2312.00863
