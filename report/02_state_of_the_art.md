# State of the Art

Project: **Foundation Model Segmentation for Robotic Scenes**  
Assignment: **Zero-Shot Segmentation Benchmark for Robotic Perception (Simulation)**  
Student id: **5884715**

This file supports the **State of the Art** section in
[`REPORT.md`](../REPORT.md). It summarizes the literature context and explains
why this project benchmarks segmentation models specifically for robotic
scenes.

For reusable figures and result tables, use
[`figures_and_tables.md`](figures_and_tables.md).

---

## 1. State-of-the-Art Taxonomy

| Research stream | Representative methods | Contribution | Robotics limitation |
|---|---|---|---|
| Promptable foundation segmentation | SAM, SAM ViT-H, SAM ViT-B | Zero-shot masks from point, box, or mask prompts | Requires a target cue; heavy variants are slow. |
| Temporal foundation segmentation | SAM2 | Extends promptable segmentation toward image/video streams | Frame-level benchmark results do not prove complete temporal stability. |
| Fast/lightweight SAM variants | FastSAM, MobileSAM, EfficientSAM | Reduce model size, latency, or encoder cost | May lose boundary quality, small-object quality, or robustness. |
| Supervised segmentation | Mask R-CNN, DeepLabV3+, YOLOv8-seg | Strong target-domain performance after training | Requires labels and is not pure zero-shot. |
| Robotic simulation and synthetic data | Isaac Sim, BlenderProc, OCID-style clutter data | Controlled annotations and targeted challenge generation | Synthetic data still has a sim-to-real gap. |
| Recent frontier beyond this benchmark | SAM3, SAM3D-style work | Concept prompting, tracking, and 3D physical structure | Outside the controlled 2D assignment scope. |

---

## 2. What Is Already Established

- Promptable segmentation can generalize to new visual settings without
  target-domain fine-tuning.
- SAM-style point and box prompts provide a flexible interface for task-guided
  segmentation.
- SAM2 moves the segment-anything idea toward continuous image/video
  perception.
- Lightweight SAM variants make promptable segmentation more plausible for
  mobile or embedded deployment.
- Supervised models remain strong when a small representative labeled dataset
  is available.
- Simulation can provide segmentation annotations for conditions that are
  expensive to collect manually.

---

## 3. What Remains Open for Robotic Perception

The literature does not fully answer whether these models are reliable as
robotic perception modules. The main open issues are:

| Open issue | Why it matters |
|---|---|
| Prompt dependency | A robot must obtain prompts from attention, detection, tracking, planning, or a human. |
| Robotic materials | Transparent and reflective objects can weaken or distort visual boundaries. |
| Small/thin parts | Screws, connectors, cables, and tools may occupy very few pixels but matter for action. |
| Occlusion and robot body | The robot itself can hide, merge with, or confuse external objects. |
| Runtime | A high-quality mask is not useful for closed-loop control if it arrives too late. |
| Sim-to-real gap | Simulation gives control and labels, but real sensors and lighting can differ. |

This project is positioned at this gap. It does not propose a new model. It
evaluates existing state-of-the-art segmentation models under robotic-scene
conditions and converts the results into deployment-oriented recommendations.

---

## 4. Model Roles in This Benchmark

| Model family | Models used | Role in the benchmark |
|---|---|---|
| Heavy zero-shot foundation models | SAM ViT-H, SAM ViT-B, SAM2 Hiera-Large, FastSAM-X | Test zero-shot promptable segmentation quality. |
| Lightweight SAM variants | MobileSAM ViT-T, EfficientSAM-Ti, EfficientSAM-S | Test edge-oriented speed-size-quality trade-offs. |
| Supervised baselines | YOLOv8-seg, Mask R-CNN, DeepLabV3+ | Test the value of small target-domain supervision. |

The comparison is intentionally not a single leaderboard. Heavy SAM models,
lightweight SAM variants, and supervised baselines answer different robotic
questions.

---

## 5. Evidence Used in the Report

| SOTA claim | Evidence from this project |
|---|---|
| Promptable models are strong but prompt-dependent. | F2, F3, T1 |
| Supervised baselines remain relevant for deployment. | F4, T3 |
| Runtime must be evaluated separately from quality. | F5, T2, T3 |
| Lightweight models require explicit trade-off analysis. | F6, T4 |
| Robotic challenge categories need separate robustness analysis. | F7, T5, E1–E6 |

Figure/table IDs refer to [`figures_and_tables.md`](figures_and_tables.md).

---

## 6. COGAR Connection

The state of the art is relevant to COGAR because segmentation is not treated
as a final product. A mask is a perceptual structure that may support:

- task-driven attention,
- object representation,
- tracking,
- planning,
- manipulation,
- uncertainty monitoring,
- and action selection.

SAM-style models are especially relevant because prompts can be interpreted as
attention signals. However, they are not complete cognitive systems: they do
not decide goals, maintain task memory, guarantee safe action, or solve prompt
generation.

---

## 7. Slide-Ready Summary

**Slide title:** State of the Art

**Main message:** Segmentation research has moved toward promptable foundation
models, but robotic deployment still requires robustness, speed, prompt
generation, and failure awareness.

**Recommended visual:** F2 zero-shot mIoU heatmap or F3 prompt-specific
winners.

---

## 8. Key References

Full source details are maintained in [`references.md`](references.md).

- SAM: Kirillov et al., 2023.
- SAM2: Ravi et al., 2024.
- FastSAM: Zhao et al., 2023.
- MobileSAM: Zhang et al., 2023.
- EfficientSAM: Xiong et al., 2023.
- Mask R-CNN: He et al., 2017.
- DeepLabV3+: Chen et al., 2018.
- Isaac Sim / BlenderProc / OCID for simulation and robotic clutter context.
