# Results Congruence and Conclusions

Project: **Foundation Model Segmentation for Robotic Scenes**  
Assignment: **Zero-Shot Segmentation Benchmark for Robotic Perception (Simulation)**  
Student id: **5884715**

This file supports the **Congruence of Results and Conclusions** section in
[`REPORT.md`](../REPORT.md). It ensures that each conclusion follows from the
available evidence and avoids overclaiming.

For reusable plots and CSV sources, use
[`figures_and_tables.md`](figures_and_tables.md).

> **Result storage:** The complete raw `results/` folder could not be included
> in Git because its predictions and checkpoints are too large. The numerical
> conclusions below are backed by committed summaries, tables, plots, and
> selected failure examples under `outputs/`; full raw results remain on the
> benchmark machine/AWS storage.

---

## 1. Main Congruence Principle

The benchmark supports **conditional recommendations**, not a universal ranking
where one model is always best.

Supported conclusion:

> Foundation segmentation models can support robotic perception, especially
> when a target prompt is available, but model choice depends on prompt mode,
> scene difficulty, runtime constraints, and downstream robotic task.

Overclaim to avoid:

> Foundation models solve robotic perception.

---

## 2. Evidence Summary

| Evidence type | Main result | Source |
|---|---|---|
| Best synthetic quality | SAM ViT-H box on BlenderProc: mIoU 0.923, boundary F1 0.905, mask AP 0.868, 0.574 FPS | T2 |
| Best robot-centered synthetic quality | SAM ViT-H box on Isaac G1: mIoU 0.752, boundary F1 0.874, mask AP 0.678, 0.574 FPS | T2 |
| Best real clutter quality | DeepLabV3+ on OCID: mIoU 0.963, boundary F1 0.880, 37.811 FPS | T2 |
| Best speed-quality trade-off | YOLOv8-seg or DeepLabV3+ depending on dataset | T3 |
| Best lightweight box trade-off | MobileSAM box: 0.883 BlenderProc, 0.693 Isaac G1, 0.824 OCID | T4 |
| Weakest challenge pattern | Robot/occlusion and small/thin structures show low weighted IoU rows | T5 |

Figure/table IDs refer to [`figures_and_tables.md`](figures_and_tables.md).

---

## 3. Hypothesis-to-Evidence Audit

| Hypothesis | Evidence status | Evidence | Congruent conclusion |
|---|---|---|---|
| H1: Prompted modes outperform automatic generation | Mostly supported, with exceptions | T1, F2, F3 | Prompted segmentation should be treated as target-guided perception; automatic mode is a harder object-discovery setting. |
| H2: Box prompts are more reliable than point prompts | Supported for strongest quality rows | T1, T2 | Box prompts are the strongest practical interface when a detector, tracker, planner, or human can provide a target region. |
| H3: Heavy SAM/SAM2 models provide high quality but low real-time feasibility | Supported | T2, F5 | Heavy SAM models are better for high-quality offline or slow deliberate perception than fast control loops. |
| H4: Lightweight variants improve deployability with quality trade-offs | Supported | T4, F6 | MobileSAM/EfficientSAM are useful only when speed, memory, and acceptable quality are considered together. |
| H5: Supervised baselines remain important for real-time robotics | Supported | T3, F4, F5 | When labeled target-domain data exists, supervised baselines can be more practical for real-time deployment. |
| H6: Robotic challenge groups reveal hidden failures | Supported | T5, T6, E1–E6, F7 | Transparent, reflective, occluded, small, thin, and robot-body cases require explicit validation. |

---

## 4. Result-to-Conclusion Rules

| Result pattern | Supported conclusion | Do not claim |
|---|---|---|
| Box prompts produce strong quality | Prompt-guided SAM-family segmentation is useful. | The robot can automatically obtain perfect prompts. |
| SAM ViT-H gives high mIoU but low FPS | Heavy SAM is suitable for high-quality/slow settings. | Heavy SAM is real-time on the tested hardware. |
| YOLOv8-seg and DeepLabV3+ reach higher FPS | Supervised baselines remain practical for robot loops. | Supervised baselines generalize as broadly as zero-shot models. |
| MobileSAM gives good box-prompt trade-offs | Lightweight SAM can support edge-oriented prompted perception. | Lightweight variants preserve full SAM quality. |
| Challenge groups expose weak cases | Robotic evaluation needs category/challenge analysis. | Average mIoU alone proves robotic reliability. |
| Failure overlays show zero-IoU cases | Qualitative failure analysis is necessary. | A few examples alone prove global model behavior. |

---

## 5. Final Recommendations

| Robotic scenario | Recommended model family | Reason |
|---|---|---|
| Highest mask quality, offline analysis, annotation support | SAM ViT-H / SAM2 with box prompts | Highest quality when latency is acceptable and a target cue exists. |
| Prompt-guided manipulation | SAM/SAM2 or MobileSAM with box prompts | Box prompts act as task-driven attention and reduce ambiguity. |
| Real-time control with target-domain labels | YOLOv8-seg or DeepLabV3+ | Higher FPS and predictable deployment after supervision. |
| Edge-oriented promptable perception | MobileSAM, sometimes EfficientSAM | Better model-size/speed trade-off than heavy SAM. |
| Open-ended object discovery | Automatic mask generation with filtering and validation | More autonomous, but slower and less controlled in clutter. |
| Transparent, reflective, occluded, small, or dynamic scenes | Any model plus extra validation | These are high-risk categories for downstream robot action. |

---

## 6. Threats to Validity

| Threat | Interpretation |
|---|---|
| Oracle prompts | Point/box prompts measure segmentation quality once a target cue exists; they do not solve prompt generation. |
| Simulation-to-real gap | Isaac/BlenderProc results are controlled evidence, not complete real-robot proof. |
| Supervised vs zero-shot settings | These are different learning regimes; final comparisons use held-out test data. |
| Metrics vs task success | mIoU/AP/FPS do not directly measure grasping, navigation, or manipulation success. |
| Runtime dependence | FPS depends on hardware, implementation, preprocessing, image size, and prompt mode. |
| Artifact storage | The complete raw `results/` folder could not be included in Git because it is too large; compact evaluated outputs and plots are committed, and full raw files remain on the benchmark machine/AWS storage. |

---

## 7. Final Conclusion

Foundation segmentation models are valuable perception modules for robotic
scene understanding, but their usefulness is conditional. Prompted SAM-family
models are appropriate when segmentation quality is the priority and a target
cue can be provided. Supervised baselines remain important when real-time
robotic control is required and labeled data is available. Lightweight SAM
variants offer a possible edge-deployment path, but only after explicit
speed-quality evaluation.

The final recommendation is therefore task-dependent: choose the segmentation
model according to quality, prompt availability, robustness, runtime, and the
downstream robotic action.

---

## 8. Slide-Ready Summary

**Slide title:** Congruence of Results and Conclusions

**Main message:** The results justify conditional model-selection rules, not a
single universal winner.

**Recommended visuals:** F2 zero-shot heatmap, F5 speed-quality scatter, F7
challenge-group plot.

---

## 9. Links

- Final report section: [`REPORT.md#5-congruence-of-results-and-conclusions`](../REPORT.md#5-congruence-of-results-and-conclusions)
- Evidence catalog: [`figures_and_tables.md`](figures_and_tables.md)
- Recommendation guide: [`../outputs/final_benchmark_assets/recommendation_guide.md`](../outputs/final_benchmark_assets/recommendation_guide.md)
