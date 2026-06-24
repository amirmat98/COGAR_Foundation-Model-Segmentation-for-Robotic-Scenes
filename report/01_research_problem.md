# Research Problem

Project: **Foundation Model Segmentation for Robotic Scenes**  
Assignment: **Zero-Shot Segmentation Benchmark for Robotic Perception (Simulation)**  
Student id: **5884715**

This file is the support page for the **Research Problem** section in
[`REPORT.md`](../REPORT.md). It defines the problem, research question,
subquestions, scope, and the evidence used to motivate the benchmark.

For reusable plots and CSV sources, use the shared catalog:
[`figures_and_tables.md`](figures_and_tables.md).

> **Evidence storage:** The complete raw `results/` folder could not be
> included in Git because it is too large. Compact evaluated evidence is
> available under `outputs/`, while the full raw results remain on the
> benchmark machine/AWS storage.

---

## 1. Problem Statement

Robots need object-level perception before they can grasp, inspect, avoid,
track, or reason about objects. In this project, segmentation is treated as the
step that converts raw visual input into object regions that can support later
robotic modules.

The problem is that general segmentation benchmarks do not fully represent the
conditions that matter in robotics. Robotic scenes include:

- transparent glass and plastic,
- reflective metal,
- small screws, connectors, cables, and thin parts,
- partial occlusion by objects or robot body parts,
- cluttered workspaces,
- moving objects and changing viewpoints,
- strict runtime constraints.

Foundation models such as SAM are attractive because they can segment new
objects without retraining. The open question is whether this zero-shot ability
is reliable enough for robotic perception, where a visual mistake can affect a
physical action.

---

## 2. Central Research Question

> To what extent can promptable foundation segmentation models provide reliable
> zero-shot object masks for robotic scene understanding in challenging
> simulated environments, and what trade-offs appear against lightweight and
> supervised alternatives in accuracy, robustness, prompt dependence, and
> real-time feasibility?

This is not a search for one universal winner. It is a model-selection problem:
the right segmentation model depends on the robotic task, prompt availability,
scene difficulty, and compute budget.

---

## 3. Research Subquestions

| ID | Subquestion | Evidence used |
|---|---|---|
| RQ1 | Which SAM-family models produce the most accurate masks under point, box, and automatic prompting? | F2, F3, T1 |
| RQ2 | Which models fail on reflective, transparent, occluded, small-part, dynamic, and robot-centered scenes? | F7, T5, T6, E1–E6 |
| RQ3 | Which models are fast enough on GPU/CPU to be plausible for robotic use? | F5, T3 |
| RQ4 | When should a robot use a heavy zero-shot model, a lightweight SAM variant, or a supervised baseline? | F4, F5, F6, T2–T4 |

Figure/table IDs refer to [`figures_and_tables.md`](figures_and_tables.md).

---

## 4. Why This Problem Matters for Robotics

In offline image segmentation, a wrong mask is usually just a lower metric
score. In robotics, a wrong mask can become a wrong object representation. That
can affect grasping, tracking, collision checking, navigation, inspection, or
human-robot interaction.

| Robotic need | Why segmentation matters |
|---|---|
| Manipulation | The robot needs object extent and boundaries for grasp planning. |
| Task-driven attention | Point and box prompts can represent where the robot should focus. |
| Scene representation | Masks provide object-level regions for reasoning and planning. |
| Tracking | Stable masks help maintain object identity over time. |
| Safety | Missing transparent, reflective, or occluded objects can cause risky actions. |
| Real-time operation | Masks must arrive fast enough for the control loop. |

---

## 5. Benchmark Scope

The project evaluates segmentation as a perception module. It does not claim to
solve the complete robotic perception-action stack.

| Included | Not claimed |
|---|---|
| Zero-shot SAM/SAM2/FastSAM/MobileSAM/EfficientSAM evaluation | Full autonomous object understanding |
| Point, box, and automatic prompt modes | Automatic generation of perfect prompts |
| Synthetic robot-centered scenes and real clutter reference | Complete real-world robot deployment proof |
| mIoU, boundary F1, mask AP, per-category IoU | Full manipulation or navigation task success |
| GPU/CPU FPS and latency | Hardware-independent real-time guarantees |
| Qualitative failure analysis | Exhaustive safety validation |

---

## 6. Slide-Ready Summary

**Slide title:** Research Problem: Robust Segmentation for Robotic Scene Understanding

**Main message:** Foundation segmentation models are promising for robots, but
robotic usefulness depends on prompt availability, robustness, runtime, and
failure awareness.

**Recommended visual:** F1 dataset examples, optionally followed by F7
challenge-group performance.

**Speaker line:**

> The research problem is not simply whether SAM can segment images. The
> problem is whether SAM-style models can become reliable perception modules
> inside robotic scenes with difficult materials, occlusions, small parts, and
> real-time constraints.

---

## 7. Links

- Final report section: [`REPORT.md#1-research-problem`](../REPORT.md#1-research-problem)
- Evidence catalog: [`figures_and_tables.md`](figures_and_tables.md)
- State of the art: [`02_state_of_the_art.md`](02_state_of_the_art.md)
