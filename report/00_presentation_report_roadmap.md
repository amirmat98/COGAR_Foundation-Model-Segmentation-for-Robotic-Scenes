# Presentation and Report Roadmap

Project: **Foundation Model Segmentation for Robotic Scenes**  
Assignment: **Zero-Shot Segmentation Benchmark for Robotic Perception (Simulation)**  
Student id: **5884715**

Recommended repo path:

```text
presentation/00_presentation_report_roadmap.md
```

This file defines the story, structure, and grading logic for the final presentation and written report. It is not meant to replace the technical benchmark files. It is meant to help the project read as a research project for **Cognitive Architectures for Robotics (COGAR)**.

---

## 1. Main Research Story

The project should not be presented only as “I benchmarked SAM models.” The stronger research story is:

> This project evaluates whether promptable foundation segmentation models can serve as reliable perception modules for embodied robotic scene understanding, especially under robotic challenges such as transparent objects, reflective surfaces, partial occlusion, small parts, dynamic objects, robot-body visibility, and real-time deployment constraints.

The project is therefore about the connection between:

1. **Perception quality**: how accurate the segmentation masks are.
2. **Robotic robustness**: whether the models survive difficult robotic-scene conditions.
3. **Cognitive architecture integration**: whether segmentation can support attention, scene representation, action selection, and the perception-action loop.
4. **Deployment feasibility**: whether the model is fast enough for GPU, CPU, or edge deployment.

---

## 2. Required Presentation Sections

The presentation must include these required elements:

1. Research Problem
2. State of the Art
3. Research Formulation
4. Cognitive Approach
5. Congruence of Results and Conclusions

The following sections should be added around them to make the presentation coherent:

6. Dataset and Simulation Design
7. Benchmark Protocol
8. Quantitative Results
9. Failure Mode Analysis
10. Final Recommendations

---

## 3. Recommended Slide Deck Structure

For a 10–12 minute presentation, use **11 slides**.

| Slide | Title | Required element covered | Purpose |
|---:|---|---|---|
| 1 | Title and Research Question | Research Problem | Introduce the project and main question. |
| 2 | Research Problem | Research Problem | Explain why segmentation is difficult in robotic scenes. |
| 3 | State of the Art | State of the Art | Summarize SAM, SAM2, FastSAM, MobileSAM, EfficientSAM, and classical baselines. |
| 4 | Research Gap | State of the Art | Explain why general image segmentation is not enough for embodied robotic perception. |
| 5 | Research Formulation | Research Formulation | State objectives, hypotheses, and methodology. |
| 6 | Dataset and Simulation | Research Formulation | Show Isaac Unitree G1, BlenderProc, OCID, and challenge groups. |
| 7 | Benchmark Protocol | Research Formulation | Explain prompt modes, zero-shot models, baselines, and metrics. |
| 8 | Quantitative Results | Congruence of Results and Conclusions | Present mIoU, boundary F1, mask AP, and FPS. |
| 9 | Failure Mode Analysis | Congruence of Results and Conclusions | Show where and why models fail. |
| 10 | Cognitive Approach | Cognitive Approach | Connect segmentation to COGAR: attention, embodiment, perception-action loop. |
| 11 | Conclusions and Recommendations | Congruence of Results and Conclusions | Match conclusions to evidence and give model-selection guidance. |

---

## 4. Recommended Report Structure

Create the final report as:

```text
report/final_report.md
```

Suggested structure:

```text
1. Introduction
2. Research Problem
3. State of the Art
4. Research Formulation
   4.1 Objectives
   4.2 Hypotheses
   4.3 Methodology
5. Cognitive Approach and COGAR Connection
6. Dataset and Simulation Design
7. Benchmark Protocol
8. Results
9. Failure Mode Analysis
10. Congruence of Results and Conclusions
11. Limitations
12. Recommendations
13. References
```

The report should be written as a research document, not as a task checklist. The task checklist can stay in `REPORT.md`; the final report should explain the scientific motivation, method, evidence, and conclusions.

---

## 5. Core Research Question

Use this as the central research question:

> Can promptable foundation segmentation models provide reliable zero-shot object masks for robotic perception in challenging simulated scenes, and how do they compare with lightweight and supervised segmentation models in terms of accuracy, robustness, and real-time feasibility?

---

## 6. Main Claim

Use this as the main claim throughout the presentation and report:

> SAM-family models are strong general-purpose segmentation modules, especially when a robot or upstream system can provide a box prompt. However, robotic perception introduces domain-specific challenges that reduce reliability, including transparent and reflective surfaces, occlusion, small parts, robot-body visibility, clutter, and dynamic scenes. Therefore, model selection should depend on the robotic use case: high-quality offline perception, prompt-guided manipulation, real-time closed-loop control, or edge deployment.

---

## 7. Key Methodological Note

The report and presentation should explicitly explain the difference between prompt modes:

- **Point prompt**: evaluates how well the model segments a known target from a simple spatial cue.
- **Box prompt**: evaluates how well the model segments a known target when a stronger spatial cue is available.
- **Automatic mask generation**: evaluates prompt-free object discovery.

Important wording:

> Point-prompt and box-prompt evaluations are oracle-prompt evaluations. The prompt is derived from ground-truth object information to isolate the segmentation quality of each model once a target cue is available. In a deployed robotic system, such prompts would need to come from an upstream module such as a detector, tracker, grasp planner, human operator, or robot task prior.

This prevents the final presentation from overclaiming that box-prompt performance is fully autonomous perception.

---

## 8. COGAR Connection

The COGAR connection should be explicit. Segmentation should be framed as a perception module inside a cognitive robotic architecture.

| COGAR concept | Connection to this project |
|---|---|
| Embodiment | The benchmark uses robot-centered scenes, including Unitree G1 simulation, instead of only generic internet images. |
| Situated perception | The model is evaluated in context: clutter, occlusion, robot body, reflective objects, transparent objects, and dynamic scenes. |
| Attention | Point and box prompts simulate task-driven visual attention. |
| Scene representation | Segmentation masks can become object-level symbols or regions for downstream reasoning. |
| Perception-action loop | Mask quality affects grasping, manipulation, tracking, and navigation decisions. |
| Real-time cognition | FPS and latency determine whether perception can run inside a robot control loop. |
| Robustness | Failure analysis identifies conditions where perception becomes unreliable for action. |

A strong sentence for the cognitive approach section:

> In this project, segmentation is treated not only as a computer vision output, but as a cognitive interface between raw visual input and robot-level decisions.

---

## 9. Evidence Discipline

The final presentation must keep the evidence and conclusions congruent.

Use strong conclusions only when directly supported by final results.

Safe conclusion style:

> The results suggest that box-prompted foundation models are effective when a target cue is available, while automatic segmentation and lightweight variants require careful selection depending on speed and robustness requirements.

Avoid overclaiming:

> Foundation models solve robotic perception.

Better wording:

> Foundation segmentation models can support robotic perception, but they do not remove the need for prompt generation, domain-specific validation, speed analysis, and failure-mode awareness.

---

## 10. File Plan for Step-by-Step Development

The following files should be created one by one:

```text
presentation/00_presentation_report_roadmap.md
presentation/01_research_problem.md
presentation/02_state_of_the_art.md
presentation/03_research_formulation.md
presentation/04_cognitive_approach.md
presentation/05_results_congruence_and_conclusions.md
presentation/06_slide_deck_outline.md
report/final_report.md
report/references.md
```

Each section file can later be reused directly in the final report and converted into slides.

---

## 11. Reference Anchors

Use these sources in the final report and slide speaker notes:

- Segment Anything: promptable segmentation, zero-shot transfer, SA-1B dataset.
  - https://arxiv.org/abs/2304.02643
- SAM2: promptable image/video segmentation with streaming memory.
  - https://arxiv.org/abs/2408.00714
- FastSAM: fast segment-anything-style model.
  - https://arxiv.org/abs/2306.12156
- MobileSAM: lightweight SAM variant for smaller deployment settings.
  - https://github.com/ChaoningZhang/MobileSAM
- EfficientSAM: efficient SAM-style model based on masked-image pretraining.
  - https://arxiv.org/abs/2312.00863
- Mask R-CNN: standard instance segmentation baseline.
  - https://arxiv.org/abs/1703.06870
- DeepLabV3+: semantic segmentation with encoder-decoder and atrous separable convolution.
  - https://arxiv.org/abs/1802.02611
- COGAR course page: cognition in humans and robots, cognitive architectures, ROS, and robotic architectures.
  - https://corsi.unige.it/en/off.f/2023/ins/66538
- Current project report.
  - https://github.com/amirmat98/COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes/blob/main/REPORT.md
