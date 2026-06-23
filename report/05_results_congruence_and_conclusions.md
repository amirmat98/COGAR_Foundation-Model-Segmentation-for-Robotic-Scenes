# 05 — Results Congruence and Conclusions

Recommended repo path:

```text
presentation/05_results_congruence_and_conclusions.md
```

This file explains how to present the results honestly and how to make sure that the final conclusions are consistent with the evidence. It is designed for both the slide deck and the final written report.

---

## 1. Purpose of This Section

The required presentation structure asks for:

> Congruence of Results and Conclusions

This means that the presentation should not only show benchmark numbers. It must also demonstrate that the final conclusions follow logically from the obtained results.

For this project, the key idea is:

> The results should support a conditional recommendation, not a universal claim that one segmentation model is always best.

The correct conclusion is not:

> Foundation models solve robotic perception.

The correct conclusion is:

> Foundation segmentation models can support robotic perception, especially when a useful prompt is available, but their reliability depends on scene conditions, prompt type, runtime constraints, and the downstream robotic task.

---

## 2. Important Evidence Discipline

The repository currently distinguishes between legacy evaluation outputs and corrected common-test evaluation outputs.

Use the following rule in the final report and slides:

> Final quantitative comparisons must be based on the corrected common held-out test protocol. Legacy full-dataset or validation results can be mentioned only as archived preliminary evidence or provenance.

This is important because the project report says that the original Tasks 1–9 run was completed, but Task 6 is being rerun under a corrected common held-out test protocol. The older zero-shot and baseline summaries remain archived, but they are not final comparative evidence.

### Safe wording for slides

> The benchmark pipeline is complete. For final comparative claims, the project uses a corrected common held-out test protocol so that zero-shot models and supervised baselines are evaluated on the same test images.

### Avoid this wording

> The legacy validation numbers prove that model X is the final best model.

### Better wording

> Preliminary and archived runs suggest that box-prompted SAM-family models are strong in mask quality, while supervised baselines are more suitable for real-time throughput. Final comparative statements should use the corrected common-test summaries.

---

## 3. What the Results Should Be Used to Answer

The results should answer five research questions.

### RQ1 — Are foundation segmentation models useful for robotic scene segmentation?

Expected conclusion style:

> Yes, SAM-family models are useful as general-purpose segmentation modules, especially in prompted settings. However, they should be treated as perception components that require validation under robotic-scene challenges.

Evidence to use:

- mIoU and boundary F1 from zero-shot models.
- Per-category IoU for object-specific robustness.
- Qualitative examples showing correct masks and failure masks.

### RQ2 — Which prompt mode is most reliable?

Expected conclusion style:

> Box prompts are expected to be more reliable than point prompts because they provide stronger spatial constraints. Automatic mask generation is more autonomous, but it is also more difficult because the model must both discover and segment objects without a target cue.

Evidence to use:

- mIoU by prompt mode.
- Mask AP by prompt mode.
- Failure examples from automatic mask generation.
- Runtime results, because automatic generation often produces many masks and can be slower.

### RQ3 — Are lightweight SAM variants good enough for edge deployment?

Expected conclusion style:

> Lightweight variants are useful when memory size and speed matter, but they should be selected according to the robotic task. A lightweight model may be acceptable for approximate object localization, but not always for precise manipulation or transparent-object segmentation.

Evidence to use:

- FPS on GPU and CPU.
- Checkpoint size.
- mIoU/FPS trade-off.
- Failure examples for small, reflective, transparent, or occluded objects.

### RQ4 — Do classical supervised baselines still matter?

Expected conclusion style:

> Yes. Supervised baselines remain important when a small labeled subset is available and real-time operation is required. They are less flexible than zero-shot foundation models, but they may be more practical for closed-loop robotic control.

Evidence to use:

- YOLOv8-seg speed.
- DeepLabV3+ semantic segmentation speed and mIoU.
- Difference between zero-shot generality and supervised domain adaptation.

### RQ5 — What are the most important robotic failure modes?

Expected conclusion style:

> The highest-risk situations are transparent and reflective objects, small or thin structures, partial occlusion, robot-body overlap, cluttered scenes, and dynamic objects. These conditions are important because they can cause wrong downstream robotic decisions.

Evidence to use:

- Failure-mode visualizations.
- Challenge-group weighted IoU.
- Per-category IoU.
- Boundary F1 for thin structures and object boundaries.

---

## 4. Result-to-Conclusion Mapping

Use the following table in the report or speaker notes.

| Result pattern | Supported conclusion | What not to overclaim |
|---|---|---|
| Box prompts produce higher mIoU than point or automatic modes | Prompt quality strongly affects segmentation quality | Do not claim the robot can automatically obtain perfect boxes |
| SAM/SAM2 achieve high mask quality in prompted settings | Foundation models are useful for prompt-guided robotic perception | Do not claim they solve autonomous perception alone |
| Automatic mask generation is slower or less stable | Prompt-free object discovery remains difficult | Do not claim automatic mode is useless in all cases |
| Supervised baselines achieve higher FPS | Classical models remain relevant for real-time control | Do not claim they generalize as broadly as foundation models |
| Lightweight SAM variants improve speed/model size | Edge deployment is possible with trade-offs | Do not claim lightweight models preserve full SAM quality |
| Transparent/reflective/occluded/small objects fail more often | Robotic scene properties must be part of model evaluation | Do not treat generic image benchmarks as sufficient for robotics |

---

## 5. Congruence With the Main Hypotheses

This section can be used directly in the final report.

### Hypothesis 1

> Prompted SAM-family models will outperform automatic mask generation in mask quality.

Congruent interpretation:

If point and box prompts outperform automatic mode, the result supports the idea that foundation segmentation models are strongest when integrated with an attention or target-selection mechanism. This matches the cognitive interpretation of prompts as task-driven attention.

If automatic mode performs competitively in some cases, the conclusion should be narrower: automatic segmentation may work in clean or high-contrast scenes, but it still requires failure analysis before being used for autonomous robotic perception.

### Hypothesis 2

> Box prompts will be more reliable than point prompts.

Congruent interpretation:

If box prompts achieve stronger mIoU, boundary F1, or AP, the conclusion is that spatially constrained prompting provides a more reliable interface for robotic perception. In a real robot, the box could come from a detector, tracker, human operator, or task prior.

The presentation must make clear that this is an oracle-prompt experiment if the box comes from ground-truth annotations.

### Hypothesis 3

> Lightweight SAM variants will improve deployability but may reduce quality.

Congruent interpretation:

If lightweight models have better speed or smaller checkpoint size but lower quality, the conclusion is that they are useful for edge deployment only when the task can tolerate reduced segmentation precision.

If a lightweight model performs surprisingly well, the conclusion should still mention the tested conditions and not generalize beyond the evaluated datasets.

### Hypothesis 4

> Classical supervised models may be more suitable for real-time robotic control when labeled data is available.

Congruent interpretation:

If YOLOv8-seg or DeepLabV3+ achieves higher FPS, the conclusion is that supervised models remain practical for closed-loop perception. This does not contradict the value of foundation models: it shows that zero-shot flexibility and real-time efficiency are different objectives.

### Hypothesis 5

> Transparent, reflective, occluded, and small objects will be major failure categories.

Congruent interpretation:

If failure examples and challenge-group metrics show weaker results on these cases, the conclusion is that robotics needs domain-specific validation. A model that performs well on generic images may still fail on physically important robotic objects.

---

## 6. Slide Version

Suggested slide title:

> Congruence of Results and Conclusions

Suggested slide bullets:

```text
- The conclusions are conditional, not universal.
- Prompted foundation models support high-quality segmentation when a target cue is available.
- Automatic segmentation is more autonomous, but less controlled and often slower.
- Supervised baselines remain important for real-time robotic control.
- Lightweight SAM variants improve deployability but require speed-quality trade-off analysis.
- Failure modes show why robotic validation is necessary: transparency, reflection, occlusion, small parts, motion.
```

Suggested final slide message:

> The benchmark supports using foundation segmentation as a cognitive perception module, but not as a complete replacement for robotic perception pipelines.

---

## 7. Report-Ready Text

The following text can be inserted into `report/final_report.md`.

### Congruence of Results and Conclusions

The conclusions of this project are formulated as conditional recommendations rather than universal rankings. This is necessary because robotic segmentation performance depends on several interacting factors: the prompt type, the scene challenge, the model family, the computational device, and the downstream robotic task.

The benchmark evaluates foundation segmentation models in point-prompt, box-prompt, and automatic mask generation modes. These modes represent different levels of task guidance. Point and box prompts isolate the quality of the segmentation model once a target cue is available, while automatic mask generation is closer to prompt-free object discovery. Therefore, strong performance under box prompts supports the use of SAM-family models in prompt-guided robotic perception, but it does not by itself prove full autonomous object discovery.

The results should be interpreted in relation to the perception-action loop. In manipulation, navigation, and tracking, a mask is not only a visual output; it becomes an input to downstream decisions. A high-quality mask can support object localization, grasp planning, or scene representation. A poor mask can produce incorrect object boundaries, merge objects, miss small parts, or confuse transparent and reflective surfaces. For this reason, the failure-mode analysis is as important as the average mIoU.

The expected overall conclusion is that foundation segmentation models are useful for robotic scene understanding when accuracy and flexibility are prioritized, especially when a robot or upstream module can provide a prompt. However, supervised baselines remain important when real-time operation is required and labeled data from the target domain is available. Lightweight SAM variants provide a possible edge-deployment path, but they should be selected based on an explicit speed-quality trade-off rather than assumed to preserve the full capability of larger SAM models.

This interpretation is congruent with the project evidence because the benchmark does not evaluate only mask quality. It also evaluates runtime, CPU/GPU feasibility, per-category behavior, and qualitative failure modes. The final recommendation is therefore task-dependent: use high-capacity SAM-family models for high-quality prompt-guided segmentation, use supervised models for real-time control when labeled data is available, and use lightweight SAM variants when deployment constraints dominate.

---

## 8. Speaker Notes

You can use this script during the presentation.

> This section is important because I do not want to overclaim the results. The project does not show that one model is always best. Instead, it shows that model choice depends on the robotic situation.
>
> When a prompt is available, especially a box prompt, SAM-family models can produce strong segmentation masks. In a cognitive robotic architecture, that prompt can be interpreted as attention: the robot or an upstream module selects the relevant region, and the segmentation model refines it into an object mask.
>
> However, automatic mask generation is a harder problem. It asks the model to discover and segment objects without a target cue. This is closer to autonomous perception, but it is less controlled and can be slower.
>
> The speed results are also important. A model with strong mIoU may not be suitable for a real-time perception-action loop if it is too slow on CPU or GPU. That is why the supervised baselines and lightweight SAM variants are not secondary; they answer a different robotics question: can the model run inside a practical robot system?
>
> Finally, the failure analysis connects the results back to robotics. Transparent objects, reflective surfaces, occlusion, small parts, and motion are not rare edge cases in robotics. They are exactly the kind of physical conditions a robot must handle. Therefore, the conclusion is that foundation segmentation is promising, but it must be integrated with prompts, validation, speed analysis, and failure awareness.

---

## 9. Recommended Figure Ideas

Use these figures if available in the repo:

```text
outputs/final_benchmark_assets/plots/zero_shot_miou_heatmap.png
outputs/final_benchmark_assets/plots/baseline_miou_bars.png
outputs/final_benchmark_assets/plots/cuda_speed_quality_scatter.png
outputs/final_benchmark_assets/plots/lightweight_sam_tradeoff_cuda.png
outputs/final_benchmark_assets/plots/challenge_group_weighted_iou.png
outputs/final_benchmark_assets/plots/zero_shot_dataset_prompt_winners.png
```

Recommended slide layout:

1. Left side: one quality plot, such as zero-shot mIoU heatmap.
2. Right side: one speed-quality plot.
3. Bottom: one-sentence conclusion.

Example bottom sentence:

> High mask quality, real-time feasibility, and robustness are not the same objective; robotic deployment requires balancing all three.

---

## 10. Final Recommendation Logic

Use the following final recommendation style.

### For high-quality offline analysis

Use:

```text
SAM ViT-H, SAM ViT-B, or SAM2 with box prompts
```

Reason:

```text
Quality is prioritized over speed, and an accurate prompt is available.
```

### For prompt-guided manipulation

Use:

```text
SAM/SAM2 with box prompts, or MobileSAM if speed and model size matter.
```

Reason:

```text
The robot can use an upstream detector, tracker, or task prior to provide the object region.
```

### For real-time control

Use:

```text
YOLOv8-seg or DeepLabV3+
```

Reason:

```text
Supervised baselines can be faster and more predictable when labeled target-domain data is available.
```

### For edge deployment

Use:

```text
MobileSAM or EfficientSAM, depending on the required speed-quality balance.
```

Reason:

```text
The deployment constraint is model size, latency, and available compute.
```

### For fully autonomous object discovery

Use with caution:

```text
Automatic mask generation
```

Reason:

```text
It does not require prompts, but it may be slower and less reliable in cluttered robotic scenes.
```

---

## 11. Limitations to State Clearly

The following limitations make the conclusions more credible.

1. **Oracle prompts**  
   Point and box prompts are derived from ground-truth annotations. This isolates segmentation quality, but does not solve prompt generation.

2. **Simulation-to-real gap**  
   Isaac and BlenderProc provide controlled robotic scenes, but real sensors, lighting, motion blur, and material properties may differ.

3. **Dataset scope**  
   The benchmark includes important robotic challenges, but it does not cover every robot, sensor, task, or environment.

4. **Runtime environment**  
   FPS depends on hardware, implementation, batch size, image resolution, and preprocessing.

5. **Model versions**  
   Results are tied to the specific model checkpoints and code versions used in the repository.

6. **Corrected test protocol**  
   Final comparative claims should be made only from corrected common-test summaries, not from archived legacy validation/full-dataset metrics.

---

## 12. Final Conclusion Paragraph

Use this as the final paragraph of the report or final slide notes.

> This project shows that foundation segmentation models can be valuable perception modules for robotic scene understanding, but their usefulness is conditional. Prompted SAM-family models are appropriate when segmentation quality is the priority and a target cue can be provided. Supervised baselines remain important when real-time robotic control is required and labeled data is available. Lightweight SAM variants offer a possible route toward edge deployment, but require explicit speed-quality evaluation. The failure analysis shows that transparent objects, reflective surfaces, occlusion, small parts, and dynamic scenes remain difficult. Therefore, the best robotic perception system is not a single universal model, but a task-dependent integration of segmentation quality, prompting strategy, runtime feasibility, and failure awareness.

---

## 13. Reference Anchors

Use these sources in the final report and slide speaker notes:

- Current project report:
  - https://github.com/amirmat98/COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes/blob/main/REPORT.md
- Task 6 evaluation protocol:
  - https://github.com/amirmat98/COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes/blob/main/docs/tasks/task6_evaluation.md
- Current recommendation guide:
  - https://github.com/amirmat98/COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes/blob/main/outputs/final_benchmark_assets/recommendation_guide.md
- Segment Anything:
  - https://arxiv.org/abs/2304.02643
- SAM2:
  - https://arxiv.org/abs/2408.00714
- FastSAM:
  - https://arxiv.org/abs/2306.12156
- MobileSAM:
  - https://github.com/ChaoningZhang/MobileSAM
- EfficientSAM:
  - https://arxiv.org/abs/2312.00863
