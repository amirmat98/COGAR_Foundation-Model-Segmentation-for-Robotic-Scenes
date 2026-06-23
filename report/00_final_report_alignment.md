# Final Report Alignment Guide

Project: **Foundation Model Segmentation for Robotic Scenes**  
Assignment: **Zero-Shot Segmentation Benchmark for Robotic Perception (Simulation)**  
Student id: **5884715**

Recommended repo path:

```text
report/00_final_report_alignment.md
```

This file explains how the existing `REPORT.md` should be used in the final submission and how it should be connected to the required presentation/report structure.

---

## 1. Status of the Existing `REPORT.md`

The current `REPORT.md` is a strong **technical benchmark report**. It already documents:

- the assignment scope,
- Tasks 1-9,
- dataset creation and validation,
- Isaac Sim and BlenderProc simulation,
- Unitree G1 robotic platform use,
- SAM-family zero-shot inference,
- classical baseline training,
- evaluation artifacts,
- inference-speed analysis,
- failure-mode analysis,
- lightweight SAM trade-off analysis,
- final benchmark plots and recommendation guide.

Therefore, it should be kept in the repository as the main technical evidence file.

However, the presentation requirements are research-oriented. They require explicit discussion of:

1. Research Problem
2. State of the Art
3. Research Formulation
4. Cognitive Approach
5. Congruence of Results and Conclusions

The existing `REPORT.md` contains much of the evidence needed for these sections, but it does not organize the story in that exact research format. For this reason, the best strategy is not to replace `REPORT.md`, but to add a research-oriented narrative layer around it.

---

## 2. Recommended Final Report Strategy

Use `REPORT.md` as:

```text
Main technical benchmark report / evidence tracker
```

Then add either:

```text
report/final_research_report.md
```

or a section at the top of `REPORT.md` called:

```text
Research Narrative Summary
```

The cleanest solution is to keep `REPORT.md` unchanged and create a separate final report file:

```text
report/final_research_report.md
```

This avoids mixing two purposes:

- `REPORT.md` = technical execution and artifact evidence.
- `report/final_research_report.md` = research story for grading and presentation.

---

## 3. Mapping Existing Evidence to Required Sections

| Required section | Evidence already in `REPORT.md` | What still needs to be written explicitly |
|---|---|---|
| Research Problem | Scope, datasets, robotic challenges, model families | Explain why segmentation is difficult for robotic perception and why zero-shot foundation models must be tested in robot-centered scenes. |
| State of the Art | SAM-family models and classical baselines are listed | Add literature context: SAM, SAM2, FastSAM, MobileSAM, EfficientSAM, Mask R-CNN, DeepLabV3+, YOLOv8-seg, synthetic data, robotic perception. |
| Research Formulation | Tasks 1-9, metrics, datasets, corrected common-test protocol | State research question, objectives, hypotheses, variables, methodology, and evaluation protocol. |
| Cognitive Approach | Inference speed, robot platform, challenge groups, failure analysis | Connect segmentation to COGAR concepts: embodiment, attention, scene representation, perception-action loop, real-time cognition. |
| Congruence of Results and Conclusions | Final plots, recommendation guide, failure analysis, corrected evaluation note | Make sure conclusions only claim what the results support. Clearly distinguish final corrected test results from legacy archived results. |

---

## 4. Recommended Top-Level Report Structure

Create:

```text
report/final_research_report.md
```

with this structure:

```text
# Final Research Report: Foundation Model Segmentation for Robotic Scenes

1. Introduction
2. Research Problem
3. State of the Art
4. Research Formulation
   4.1 Research Question
   4.2 Objectives
   4.3 Hypotheses
   4.4 Methodology
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

Keep the existing `REPORT.md` as a linked source:

```markdown
For the detailed benchmark artifact tracker, see [`REPORT.md`](../REPORT.md).
```

---

## 5. How to Reuse the Existing `REPORT.md`

The existing report can be reused in the final research report as follows.

### Dataset and Simulation Design

Use the existing dataset table directly:

- Isaac official Unitree G1: 1000 synthetic images.
- BlenderProc COGAR-SimRobotics-1000: 1000 synthetic images.
- OCID: 2390 real RGB-D clutter frames.

This supports the claim that the benchmark covers both synthetic robotic simulation and real-world clutter validation.

### Benchmark Protocol

Use the current task summary:

- SAM ViT-H, SAM ViT-B, SAM2, and FastSAM for zero-shot inference.
- Point, box, and automatic prompts.
- YOLOv8-seg, Mask R-CNN, and DeepLabV3+ as supervised baselines.
- MobileSAM and EfficientSAM as lightweight SAM variants.
- Evaluation with mIoU, boundary F1, mask AP, per-category IoU, speed, and qualitative failures.

### Results and Conclusions

Use the final benchmark plots and recommendation guide, but be careful with wording:

- If corrected common-test summaries are complete, use them as final comparative evidence.
- If corrected common-test summaries are still pending, call previous plots and tables **legacy evidence** or **preliminary evidence**.
- Do not claim final ranking across all model families unless the corrected common-test protocol is complete.

---

## 6. Important Wording for Final Report

Use this sentence in the introduction:

> This project evaluates whether promptable foundation segmentation models can serve as reliable perception modules for embodied robotic scene understanding, especially under robotic challenges such as transparent objects, reflective surfaces, partial occlusion, small parts, dynamic objects, robot-body visibility, and real-time deployment constraints.

Use this sentence in the methodology:

> Point-prompt and box-prompt evaluations are oracle-prompt evaluations: the prompt is derived from ground-truth object information to isolate the segmentation quality of each model once a target cue is available. In a deployed robotic system, such prompts would need to come from an upstream module such as a detector, tracker, grasp planner, human operator, or robot task prior.

Use this sentence in the cognitive approach section:

> In this project, segmentation is treated not only as a computer vision output, but as a cognitive interface between raw visual input and robot-level decisions.

Use this sentence in the conclusion:

> Foundation segmentation models can support robotic perception, but they do not remove the need for prompt generation, domain-specific validation, inference-speed analysis, and failure-mode awareness.

---

## 7. Congruence Rule for the Final Defense

The final defense should follow this logic:

```text
If the result is from the corrected common held-out test protocol:
    It can be used as final comparative evidence.

If the result is from an older validation/full-dataset run:
    It can be shown only as archived or preliminary evidence.

If a conclusion depends on speed:
    Support it with FPS/latency results.

If a conclusion depends on robustness:
    Support it with challenge-group or failure-analysis evidence.

If a conclusion depends on model quality:
    Support it with mIoU, boundary F1, mask AP, or per-category IoU.
```

This prevents overclaiming and directly satisfies the required section **Congruence of Results and Conclusions**.

---

## 8. Suggested Repo Layout After Adding Research Files

```text
REPORT.md                                  # existing technical benchmark report
presentation/
  00_presentation_report_roadmap.md
  01_research_problem.md
  02_state_of_the_art.md
  03_research_formulation.md
  04_cognitive_approach.md
  05_results_congruence_and_conclusions.md
  06_slide_deck_outline.md
report/
  00_final_report_alignment.md
  final_research_report.md
  references.md
```

---

## 9. Reference Anchors

Use these references in the final research report:

- Segment Anything: promptable zero-shot segmentation and SA-1B dataset.  
  https://arxiv.org/abs/2304.02643

- SAM2: promptable image/video segmentation with streaming memory.  
  https://arxiv.org/abs/2408.00714

- COGAR course description: cognitive architectures for robots that perceive, represent knowledge, reason, and act effectively.  
  https://corsi.unige.it/en/off.f/2026/ins/93610

- Existing project technical report.  
  `REPORT.md`

---

## 10. Final Recommendation

Do not replace the current `REPORT.md`.

Instead:

1. Keep `REPORT.md` as the evidence tracker.
2. Add `report/00_final_report_alignment.md`.
3. Add `report/final_research_report.md` as the polished research-style report.
4. Build the presentation from the `presentation/*.md` files.

This gives the examiner both things they need:

- proof that the benchmark was implemented,
- a clear research narrative connected to COGAR.
