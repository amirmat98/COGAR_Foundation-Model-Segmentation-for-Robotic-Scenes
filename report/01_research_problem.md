# 01 — Research Problem

Project: **Foundation Model Segmentation for Robotic Scenes**  
Assignment: **Zero-Shot Segmentation Benchmark for Robotic Perception (Simulation)**  
Student id: **5884715**

Recommended repo path:

```text
presentation/01_research_problem.md
```

This file drafts the **Research Problem** section for the final presentation and report. It can be reused in:

```text
report/final_report.md
report/06_slide_deck_outline.md
```

---

## 1. Short Version for the Slide

Robots need reliable object-level perception to interact with the world. However, robotic scenes are harder than normal image-segmentation benchmarks because they contain transparent objects, reflective surfaces, partial occlusions, small parts, moving objects, clutter, and sometimes the robot body itself.

The research problem is therefore:

> Can zero-shot foundation segmentation models provide reliable object masks for robotic perception under realistic robotic-scene challenges, without requiring full task-specific retraining?

This project studies that problem by benchmarking SAM-family models and lightweight variants against classical supervised segmentation baselines on simulated and real robotic perception datasets.

---

## 2. Full Research Problem for the Report

Robotic perception is not only about detecting pixels in an image. For an embodied robot, perception is the first stage of a larger cognitive and behavioral process. A robot must transform raw sensory input into object-level information that can support attention, scene representation, manipulation, navigation, tracking, and action selection. In this context, segmentation masks are important because they define where objects are, which pixels belong to them, and how a robot could reason about them as separate entities.

Foundation segmentation models such as Segment Anything Model (SAM) have changed the landscape of image segmentation. SAM introduced a promptable segmentation model trained on a very large segmentation dataset, SA-1B, containing over one billion masks on eleven million images. The model is designed to transfer zero-shot to new image distributions and tasks using prompts such as points or boxes. This makes SAM-style models attractive for robotics because robots often operate in open-ended environments where collecting dense task-specific labels is expensive.

However, robotic scenes create failure cases that are not fully captured by generic internet-image segmentation. A robot may need to segment transparent glass, reflective metal, small screws, thin connectors, partially occluded objects, cluttered tabletop scenes, objects close to the robot body, or dynamic objects in motion. These cases are difficult because the visual evidence may be ambiguous, object boundaries may be weak, and the segmentation mask may directly affect a downstream action such as grasping or manipulation.

This project addresses the gap between general-purpose zero-shot segmentation and embodied robotic perception. The goal is not only to ask which model gives the highest segmentation score, but also to ask whether these models are robust enough to become useful modules inside a robotic cognitive architecture. A segmentation model that is accurate but too slow may not be suitable for closed-loop control. A model that works on opaque objects but fails on transparent or reflective objects may be unsafe for manipulation. A model that needs strong prompts may be useful only if another module can provide those prompts.

Therefore, the research problem can be formulated as follows:

> To what extent can promptable foundation segmentation models support reliable robotic scene understanding under simulation-generated robotic challenges, and what trade-offs appear between segmentation quality, robustness, prompt dependence, and real-time feasibility?

---

## 3. Why This Problem Matters in Robotics

In classical computer vision, segmentation is often evaluated as a standalone prediction problem. In robotics, segmentation is part of a perception-action loop. A wrong mask can lead to a wrong grasp, a wrong object representation, poor tracking, or incorrect action selection. The consequences of segmentation errors are therefore more important in robotics than in purely offline image analysis.

This is especially relevant for cognitive robotics. The COGAR perspective treats perception as a component of an embodied intelligent system, not as an isolated image-processing task. In such a system, visual segmentation can act as an interface between raw pixels and higher-level cognitive functions such as attention, memory, planning, and decision-making.

For example:

| Robotic need | Why segmentation matters |
|---|---|
| Object manipulation | The robot needs a clean object mask to estimate where to grasp or interact. |
| Task-driven attention | Point and box prompts can represent a robot focusing on a relevant target. |
| Scene representation | Masks can become object regions for downstream reasoning. |
| Tracking | A stable mask helps follow an object across time. |
| Safety | Failures on transparent, reflective, or occluded objects can produce risky actions. |
| Real-time behavior | A segmentation method must be fast enough for the robot control loop. |

---

## 4. Specific Robotic Challenges Addressed

This project focuses on robotic-scene challenges that are especially relevant for embodied perception:

### 4.1 Transparent Objects

Transparent objects are difficult because their appearance depends heavily on background, lighting, and refraction. They may also cause missing or unreliable depth information in RGB-D perception. In manipulation settings, this makes perception of glass, plastic containers, and other transparent objects a known challenge.

### 4.2 Reflective Surfaces

Reflective metal and glossy objects can confuse segmentation models because boundaries and textures may be distorted by reflections. A model may segment the reflected pattern instead of the physical object.

### 4.3 Partial Occlusion

Robots often see objects partially hidden behind other objects, tools, hands, or their own body. Partial occlusion makes it harder to recover full object extent and can split one object into multiple predicted masks.

### 4.4 Small Parts and Thin Structures

Robotic tasks often involve screws, connectors, cables, grippers, handles, and thin object parts. These regions are easy to miss, but they may be important for manipulation.

### 4.5 Dynamic or Moving Objects

A robot operating in a real environment may observe moving objects or changing scene layouts. This creates a need for segmentation models that are not only accurate on single images but also useful in time-dependent perception.

### 4.6 Robot-Centered Scenes

Unlike generic segmentation datasets, robotic scenes may include the robot itself, close-range objects, unusual camera viewpoints, and task-specific object layouts. This project uses robot-centered simulation to test segmentation under conditions closer to robotic deployment.

---

## 5. Research Gap

The main research gap is the difference between general zero-shot segmentation ability and robotic usefulness.

SAM-style models are designed to segment many kinds of objects in many kinds of images. But a robot needs more than general image segmentation. It needs segmentation that is:

1. **Robust** under difficult materials such as glass and metal.
2. **Reliable** under occlusion and clutter.
3. **Sensitive** to small and thin task-relevant parts.
4. **Compatible** with task-driven attention and prompt generation.
5. **Fast enough** for real-time or near-real-time robotic control.
6. **Interpretable enough** for failure analysis and model selection.

This gap motivates a benchmark that evaluates not only average mask quality, but also challenge-specific failure modes and deployment trade-offs.

---

## 6. How This Project Addresses the Problem

The project addresses the research problem by creating and evaluating a robotic segmentation benchmark with three complementary dataset sources:

| Dataset | Role in the problem |
|---|---|
| Isaac official Unitree G1 dataset | Main robot-centered simulation benchmark with Unitree G1 scenes. |
| BlenderProc COGAR-SimRobotics-1000 | Secondary simulation benchmark with controlled challenge variation. |
| OCID | Real-world cluttered-object dataset for robustness and domain-gap analysis. |

The benchmark evaluates:

- Zero-shot foundation segmentation models: SAM ViT-H, SAM ViT-B, SAM2, and FastSAM.
- Lightweight SAM variants: MobileSAM and EfficientSAM.
- Classical supervised baselines: YOLOv8-seg, Mask R-CNN, and DeepLabV3+.
- Prompting strategies: point prompts, box prompts, and automatic mask generation.
- Metrics: mIoU, boundary F1, mask AP, per-category IoU, inference FPS, and qualitative failure modes.

This design allows the project to compare not only which model performs best, but also which model is most appropriate for different robotic situations.

---

## 7. Research Problem as Slide Speaker Notes

Use this as the spoken explanation for the Research Problem slide:

> The problem I address is not simply object segmentation in images. In robotics, segmentation is part of the perception-action loop. If the robot receives a wrong object mask, this can affect grasping, tracking, navigation, and decision-making. Foundation models such as SAM are attractive because they can segment objects without retraining, but robotic scenes contain difficult cases such as transparent glass, reflective metal, partial occlusion, small connectors, moving objects, and the robot body itself. Therefore, my research question is whether zero-shot foundation segmentation models are reliable enough to support robotic scene understanding, and how their accuracy, robustness, and speed compare with lightweight and supervised alternatives.

---

## 8. Suggested Slide Content

### Slide Title

**Research Problem: Robust Segmentation for Robotic Scene Understanding**

### Slide Bullets

- Robots need object-level perception for manipulation, tracking, and action selection.
- Generic image segmentation is not enough for robotic deployment.
- Robotic scenes contain difficult cases:
  - transparent glass,
  - reflective metal,
  - occlusions,
  - small screws/connectors,
  - moving objects,
  - robot-body visibility.
- Foundation models promise zero-shot segmentation, but their robotic reliability must be tested.
- The key question: **Which segmentation model is reliable enough, fast enough, and robust enough for robotic perception?**

### Suggested Visual

Use one image grid from the project dataset:

```text
outputs/final_benchmark_assets/plots/dataset_examples.png
```

If available, annotate the visual with labels such as:

```text
transparent | reflective | occluded | small parts | robot-centered | dynamic
```

---

## 9. Transition to the Next Section

End this section with:

> To understand why this problem is timely, the next section reviews the state of the art in promptable segmentation, lightweight foundation segmentation, and classical supervised segmentation baselines.

This transition leads directly into:

```text
presentation/02_state_of_the_art.md
```

---

## 10. References for This Section

- Kirillov, A. et al. Segment Anything. arXiv:2304.02643.  
  https://arxiv.org/abs/2304.02643

- Ravi, N. et al. SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714.  
  https://arxiv.org/abs/2408.00714

- Jiang, J. et al. Robotic Perception of Transparent Objects: A Review. arXiv:2304.00157.  
  https://arxiv.org/abs/2304.00157

- University of Genoa. Cognitive Architectures for Robotics course page.  
  https://corsi.unige.it/en/off.f/2026/ins/93610

- Current project report.  
  https://github.com/amirmat98/COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes/blob/main/REPORT.md
