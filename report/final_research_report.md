# Final Research Report

**Project:** Foundation Model Segmentation for Robotic Scenes  
**Assignment:** Zero-Shot Segmentation Benchmark for Robotic Perception (Simulation)  
**Student id:** 5884715  
**Recommended repo path:** `report/final_research_report.md`

---

## Abstract

Robotic perception requires object-level scene understanding that is accurate, robust, and fast enough to support action. Recent foundation segmentation models such as Segment Anything (SAM) and SAM2 provide promptable zero-shot segmentation, but robotic environments introduce specific challenges that are not fully captured by generic image benchmarks: reflective metal, transparent glass, partial occlusion, small parts, robot-body visibility, clutter, and dynamic objects. This project investigates whether foundation segmentation models can operate as reliable perception modules for robotic scene understanding under such conditions.

The benchmark evaluates SAM ViT-H, SAM ViT-B, SAM2, FastSAM, MobileSAM, and EfficientSAM in zero-shot settings using point prompts, box prompts, and automatic mask generation where applicable. It compares these models with supervised baselines trained on small subsets: YOLOv8-seg, Mask R-CNN, and DeepLabV3+. The project uses two generated synthetic datasets, Isaac official Unitree G1 and BlenderProc COGAR-SimRobotics-1000, plus OCID as a real-world clutter dataset for robustness and domain-gap analysis. The evaluation considers segmentation quality, boundary accuracy, mask average precision, per-category behavior, inference speed on GPU and CPU, lightweight deployment trade-offs, and qualitative failure modes.

The research contribution is not only a technical benchmark, but also a cognitive-robotics analysis of how segmentation models can support embodied perception. In a cognitive robotic architecture, segmentation masks can act as an interface between raw visual input and object-level reasoning, attention, scene representation, tracking, grasping, and action selection. The final conclusion is conditional: foundation segmentation models can support robotic perception when reliable prompts or object priors are available, but they do not remove the need for domain-specific validation, speed analysis, prompt generation, and failure-mode awareness.

---

## 1. Introduction

Robots acting in real environments must perceive objects, surfaces, obstacles, tools, and task-relevant regions before they can reason or act. Object segmentation is therefore a central component of robotic perception. A segmentation mask can define where an object is, how it is shaped, whether it is graspable, whether it is occluded, and whether it should be tracked over time. In manipulation, mobile robotics, and human-robot interaction, the quality of segmentation can directly affect downstream decisions.

The recent emergence of foundation models for segmentation has changed the computer vision landscape. SAM introduced a promptable segmentation model trained for zero-shot transfer to new image distributions and tasks. SAM2 extended this idea to both images and videos with a transformer architecture and streaming memory. These models are attractive for robotics because robots often face new scenes, unknown objects, and changing environments where collecting task-specific labels is expensive.

However, robotic perception is not equivalent to generic image segmentation. Robotic scenes often include transparent objects, reflective metal, small screws and connectors, motion blur, partial occlusions, sensor artifacts, robot-body regions, and cluttered interaction spaces. A model that performs well on general benchmarks may still fail in a robot-centered environment. In addition, robots often require real-time inference, especially when perception is part of a closed-loop control system.

This project addresses this gap by building and evaluating a systematic benchmark for zero-shot foundation segmentation in robotic scenes. The benchmark compares heavy foundation models, lightweight SAM variants, and supervised baselines under the same robotic-scene challenges. The report connects the technical evaluation to the Cognitive Architectures for Robotics (COGAR) perspective by treating segmentation as a perception module inside a broader embodied cognitive architecture.

---

## 2. Research Problem

The research problem is:

> Can promptable foundation segmentation models provide reliable zero-shot object masks for robotic perception in challenging simulated scenes, and how do they compare with lightweight and supervised segmentation models in terms of accuracy, robustness, and real-time feasibility?

This problem has three dimensions.

First, the perception dimension asks whether the model can produce accurate masks for robotic objects and scene elements. This is measured using metrics such as mean Intersection over Union (mIoU), boundary F1, mask AP, and per-category IoU.

Second, the robustness dimension asks whether the model remains reliable under robotic-scene challenges. These include reflective metal, transparent glass, partial occlusion, small parts, dynamic objects, clutter, and robot-body visibility. These conditions matter because errors in such cases can cause downstream failures in manipulation, navigation, or tracking.

Third, the deployment dimension asks whether the model can run fast enough for practical robotic use. A high-quality segmentation model may be useful for offline analysis or high-level planning, but not for real-time control if its inference speed is too low. For this reason, the benchmark includes GPU and CPU speed measurements and explicitly evaluates lightweight variants such as MobileSAM and EfficientSAM.

The key challenge is therefore not simply identifying the most accurate model. The challenge is to understand which model is appropriate for which robotic scenario: prompt-guided manipulation, automatic object discovery, real-time closed-loop control, CPU-only execution, or edge deployment.

---

## 3. State of the Art

### 3.1 Promptable foundation segmentation

SAM introduced the idea of a general promptable segmentation model. It accepts prompts such as points and boxes and produces object masks. Its design goal is zero-shot transfer, meaning that it can be applied to new image distributions and segmentation tasks without task-specific retraining. This makes SAM relevant to robotics, where a robot may encounter unfamiliar objects and environments.

SAM2 extends promptable segmentation from images to videos. Its streaming-memory design is especially relevant to robotic perception because robots perceive scenes over time. Dynamic scenes, moving objects, and temporary occlusions require more than a single-image interpretation. A video-aware segmentation model can potentially support tracking and temporal consistency, which are important for embodied systems.

### 3.2 Efficient and lightweight SAM variants

Although SAM and SAM2 are powerful, their computational requirements can be high. This creates a problem for robots with limited compute, especially mobile robots, embedded platforms, or systems that need real-time feedback.

FastSAM addresses the speed issue by reformulating the segment-anything task using a fast instance-segmentation pipeline followed by prompt-guided mask selection. MobileSAM replaces SAM's heavy image encoder with a lightweight encoder while preserving compatibility with the SAM-style prompt-and-mask interface. EfficientSAM uses masked-image pretraining to build lightweight SAM-style models with reduced complexity.

These models are important because robotic perception is constrained by speed, memory, power, and latency. A model with slightly lower quality may be preferable if it enables real-time operation on the target robot.

### 3.3 Classical supervised baselines

The benchmark also includes classical supervised segmentation baselines. Mask R-CNN is a standard instance segmentation architecture that detects object instances and predicts a mask for each instance. DeepLabV3+ is a semantic segmentation architecture that combines multi-scale context with decoder-based boundary refinement. YOLOv8-seg is a real-time instance segmentation model from the YOLO family.

These baselines serve an important role. Foundation models are attractive because of their zero-shot ability, but supervised models can still be more efficient or more reliable when a small amount of labeled domain-specific data is available. Comparing foundation models with supervised baselines prevents the evaluation from assuming that zero-shot models are automatically the best choice for robotics.

### 3.4 Robotic perception gap

The state of the art in segmentation is strong, but a gap remains between general image segmentation and robotic perception. Robotic environments are embodied, situated, and action-oriented. A segmentation error is not only a visual error; it can affect grasping, collision avoidance, tracking, and planning. Transparent and reflective objects are particularly difficult because their visual appearance can violate assumptions about texture, color, and boundaries. Small parts such as screws and connectors are difficult because they occupy few pixels and often appear in cluttered scenes. Occlusion is difficult because a robot must decide whether a partially visible region belongs to the same object or to the background.

This project is positioned in that gap: it evaluates foundation segmentation models not as generic vision tools, but as candidate components of robotic cognitive architectures.

---

## 4. Research Formulation

### 4.1 Objectives

The objectives of the project are:

1. Create or curate robotic-scene datasets in simulation with segmentation annotations and robotics-specific challenges.
2. Use robotic simulation environments to generate scenes containing reflective, transparent, occluded, small, and dynamic objects.
3. Use a simulated robotic platform, especially Unitree G1, to make the scenes robot-centered rather than generic.
4. Evaluate SAM-family models in zero-shot mode using point prompts, box prompts, and automatic mask generation.
5. Compare zero-shot foundation models with supervised baselines trained on small subsets.
6. Evaluate models using standard segmentation metrics.
7. Measure inference speed on GPU and CPU.
8. Analyze qualitative failure modes.
9. Evaluate lightweight SAM variants for edge-deployment trade-offs.
10. Produce final recommendations for choosing segmentation models in different robotic scenarios.

### 4.2 Hypotheses

The benchmark is guided by the following hypotheses:

**H1: Prompted foundation models will outperform automatic mask generation in mask quality.**  
Point and box prompts provide a target cue, so the model does not need to discover all objects in the scene from scratch.

**H2: Box prompts will generally be more reliable than point prompts.**  
A box prompt gives stronger spatial information than a single point and can reduce ambiguity in cluttered scenes.

**H3: Lightweight SAM variants will improve deployment feasibility but may reduce quality.**  
MobileSAM and EfficientSAM are expected to offer useful trade-offs, but they may struggle with difficult boundaries, small objects, or clutter.

**H4: Supervised baselines may be more suitable for real-time robotic loops when labeled target-domain data is available.**  
Models such as YOLOv8-seg and DeepLabV3+ can be faster and more predictable after fine-tuning on the target domain.

**H5: Transparent, reflective, occluded, and small objects will form the main failure categories.**  
These categories are known to be difficult for visual perception and are especially important in robotics.

### 4.3 Methodology

The project follows a benchmark methodology:

1. Generate and organize datasets.
2. Define prompt generation protocols.
3. Run zero-shot foundation models.
4. Train supervised baselines on small subsets.
5. Evaluate all models with segmentation metrics.
6. Measure inference speed.
7. Analyze qualitative failures.
8. Combine results into recommendations.

The project uses three datasets:

| Dataset | Type | Purpose |
|---|---|---|
| Isaac official Unitree G1 | Synthetic, Isaac Sim | Main robot-centered simulation benchmark |
| BlenderProc COGAR-SimRobotics-1000 | Synthetic, BlenderProc | Secondary synthetic benchmark with controlled challenge coverage |
| OCID | Real RGB-D clutter dataset | Real-world robustness and domain-gap reference |

The technical report records that the Isaac dataset contains 1000 images with 72,695 COCO annotations and 16 categories, while the BlenderProc dataset contains 1000 images and 8,768 COCO annotations. OCID is used as an external real-world clutter dataset.

---

## 5. Cognitive Approach and COGAR Connection

The COGAR connection is central to the interpretation of the project. The benchmark is not only a comparison of segmentation networks. It studies whether foundation segmentation can act as a perception component in a cognitive robotic architecture.

A cognitive robotic system must transform raw sensory data into representations that can support reasoning and action. In this project, segmentation masks are treated as an intermediate cognitive representation. They are more structured than raw pixels but still grounded in visual perception. A mask can define an object region, guide attention, support tracking, and provide input to downstream modules such as grasp planning or object reasoning.

The cognitive interpretation can be summarized as follows:

| COGAR concept | Connection to the project |
|---|---|
| Embodiment | The benchmark uses robot-centered scenes, including Unitree G1 simulation. |
| Situated perception | Models are evaluated in clutter, occlusion, reflection, transparency, and robot-body contexts. |
| Attention | Point and box prompts simulate task-driven visual attention. |
| Scene representation | Masks can become object-level regions for downstream reasoning. |
| Perception-action loop | Mask quality affects manipulation, tracking, navigation, and planning. |
| Real-time cognition | FPS and latency determine whether segmentation can run inside a control loop. |
| Robustness | Failure analysis identifies conditions where perception becomes unreliable for action. |

The prompt modes are especially important from a cognitive perspective. A point prompt can represent a minimal attentional cue. A box prompt can represent a stronger task prior, such as a detection result or a region selected by a planner. Automatic mask generation represents open-ended scene parsing, where the system must discover object regions without a target cue.

This framing prevents the project from being interpreted as a simple computer vision benchmark. The deeper research question is whether foundation segmentation can support the perception layer of an embodied cognitive architecture.

---

## 6. Dataset and Simulation Design

The project uses simulation because simulation enables systematic control over robotic-scene conditions. In a real laboratory, collecting 500 or more annotated images with transparent objects, reflective surfaces, small parts, partial occlusions, moving objects, and robot interaction contexts would be expensive. Simulation allows the project to generate scenes with known segmentation masks and controlled challenge categories.

### 6.1 Isaac official Unitree G1 dataset

The Isaac dataset is the main robotic simulation dataset. It uses Isaac Sim and the official Unitree G1 asset to create robot-centered scenes. This is important because the benchmark is about robotic perception, not only object segmentation in generic images. The robot body and camera perspective affect what appears in the scene and how objects are occluded.

The dataset contains 1000 images. According to the technical report, its COCO export contains 72,695 annotations and 16 categories. The challenge coverage includes reflective metal, transparent glass, partial occlusion, small parts, dynamic objects, and robot close-range scenes.

### 6.2 BlenderProc COGAR-SimRobotics-1000

The BlenderProc dataset is a secondary synthetic dataset. It provides controlled simulation diversity and complements the Isaac dataset. It also contains 1000 images. According to the technical report, the full dataset contains 8,768 COCO annotations after normalization and export.

### 6.3 OCID

OCID is used as a real-world clutter dataset. It is not generated by this project and is not re-hosted. Its role is to test how conclusions from simulation relate to real RGB-D clutter scenes. This helps address the domain gap between synthetic scenes and real robotic perception.

---

## 7. Benchmark Protocol

### 7.1 Zero-shot foundation models

The zero-shot benchmark evaluates:

- SAM ViT-H
- SAM ViT-B
- SAM2
- FastSAM
- MobileSAM
- EfficientSAM

The main zero-shot prompt modes are:

- Point prompt
- Box prompt
- Automatic mask generation

Point-prompt and box-prompt results must be interpreted carefully. These are oracle-prompt evaluations: the prompt is derived from ground-truth object information to isolate the segmentation ability of each model once a target cue is available. In a deployed robot, the cue would need to come from a detector, tracker, robot task prior, grasp planner, or human operator.

Automatic mask generation is the closest benchmark setting to prompt-free object discovery. However, it is usually more computationally expensive and can be less controlled in cluttered scenes.

### 7.2 Supervised baselines

The supervised baselines are:

- YOLOv8-seg
- Mask R-CNN
- DeepLabV3+

These models are trained on small subsets and evaluated as classical comparison points. The role of these baselines is to test whether a small amount of target-domain supervision can outperform or complement zero-shot foundation segmentation.

### 7.3 Evaluation metrics

The benchmark uses:

- mIoU
- Boundary F1
- Mask AP / AP50 / AP75
- Per-category IoU
- Challenge-group performance
- GPU and CPU inference speed
- Qualitative failure analysis

mIoU measures mask overlap. Boundary F1 measures contour quality, which matters for manipulation and grasping. Mask AP measures instance-level detection and segmentation quality. Per-category IoU and challenge-group analysis identify which object types or scene conditions are difficult. FPS and latency determine whether the model can support real-time robotic use.

### 7.4 Corrected common-test protocol

The technical report states that the original Tasks 1-9 run was completed on 2026-06-17, but that Task 6 is being rerun under a corrected common held-out test protocol. Legacy validation and full-dataset metrics remain archived but are not final comparative evidence.

For this reason, final numerical conclusions should be drawn from the corrected common-test summaries:

- `outputs/task6_evaluation/zero_shot/test/summary.csv`
- `outputs/task6_evaluation/baselines/test/summary.csv`

If those files are not yet populated, this report should avoid presenting legacy numbers as final model rankings. Legacy outputs can still be discussed as provenance or preliminary evidence, but they should not be used as the final basis for comparative claims.

---

## 8. Results

This section should be updated using the corrected common-test evaluation summaries. The technical report already identifies where the corrected summaries are written:

- Zero-shot corrected summary: `outputs/task6_evaluation/zero_shot/test/summary.csv`
- Baseline corrected summary: `outputs/task6_evaluation/baselines/test/summary.csv`

The following structure should be used for the final results.

### 8.1 Segmentation quality

The first result to present is segmentation quality across datasets, models, and prompt modes. The main comparison should report mIoU, boundary F1, and mask AP. The expected interpretation is not simply “which model wins,” but which prompt/model combination is appropriate for a robotic perception setting.

Recommended discussion points:

- Compare point, box, and automatic modes.
- Identify whether box prompting is consistently stronger than point prompting.
- Compare SAM/SAM2/FastSAM on Isaac, BlenderProc, and OCID.
- Identify if the best model differs by dataset.
- Discuss whether synthetic and real-world datasets show different behavior.

### 8.2 Supervised baselines

The second result should compare zero-shot foundation models with supervised baselines. The baselines are trained on small subsets, so they are not zero-shot, but they represent practical robotic deployment when limited labeled data is available.

Recommended discussion points:

- Do supervised baselines outperform zero-shot models on any dataset?
- Are baselines faster than foundation models?
- Is the gain from supervision worth the labeling cost?
- Which baseline is most practical for real-time robotic use?

### 8.3 Inference speed

The speed analysis should report GPU and CPU performance. It should not be treated as a secondary detail because real-time feasibility is central to robotics.

Recommended discussion points:

- Which models are feasible for real-time GPU use?
- Which models are too slow for closed-loop control?
- Which models remain usable on CPU?
- How do automatic mask generation modes compare with prompted modes?
- What is the speed-quality trade-off?

### 8.4 Lightweight SAM variants

MobileSAM and EfficientSAM should be discussed as edge-deployment candidates. Their value is not only whether they beat SAM ViT-H in accuracy. Their value is whether they provide acceptable masks at lower computational cost.

Recommended discussion points:

- Does MobileSAM provide a strong speed-size trade-off?
- Does EfficientSAM provide better quality in some prompt modes?
- Are lightweight variants suitable for CPU-only use?
- Which failure modes become worse when the model is compressed?

### 8.5 Robotic challenge groups

Challenge-group results should be used to connect quantitative performance with robotic relevance. The most important groups are:

- Transparent and reflective objects
- Small parts and thin structures
- Robot body and occlusion
- Dynamic or moving objects
- Cluttered scene regions

A model should not be recommended for robotic deployment only because it has high average mIoU. If it fails on high-risk robotic categories, this limitation must be stated.

---

## 9. Failure Mode Analysis

Failure analysis is essential because average metrics can hide dangerous robotic failures. A robot does not only need a good average segmentation score; it needs reliability in the situations that matter for action.

The project identifies failure modes through qualitative examples and challenge-group summaries. The technical report records that failure analysis produced 10 representative failure visualizations and 151 challenge-group rows.

The main expected failure categories are:

### 9.1 Transparent objects

Transparent objects are difficult because their boundaries may be defined by reflections, refraction, or background texture rather than direct color contrast. A segmentation model may merge transparent objects with the background or miss them entirely.

### 9.2 Reflective surfaces

Reflective metal can create highlights and mirror-like regions that confuse boundary detection. The model may segment reflections instead of physical object boundaries.

### 9.3 Small parts and thin structures

Small screws, connectors, and thin elements occupy few pixels. They can be missed by models that operate at lower resolution or that prefer larger coherent regions.

### 9.4 Occlusion and robot-body visibility

Partial occlusion creates ambiguity about object extent. Robot-body regions can also confuse the model when robot limbs or grippers overlap task objects.

### 9.5 Dynamic objects

Moving objects introduce temporal difficulty. Single-image segmentation may be insufficient when the scene changes over time. SAM2 is relevant here because it is designed for image and video segmentation, but dynamic robotic scenes still require validation.

The qualitative conclusion should be that foundation segmentation models are useful but not fail-safe. For robotic use, mask outputs should be validated by downstream checks, temporal consistency, depth information, uncertainty estimates, or task-specific constraints.

---

## 10. Congruence of Results and Conclusions

The final conclusions must match the strength of the evidence. The project should avoid overclaiming that foundation segmentation solves robotic perception. The results support a more precise claim:

> Foundation segmentation models can support robotic perception, especially when prompt cues are available, but they require prompt generation, domain-specific validation, speed analysis, and failure-mode awareness before deployment.

The conclusion should be congruent with the results in the following ways.

### 10.1 If box prompts perform best

The correct conclusion is not that the model is fully autonomous. The correct conclusion is:

> Box-prompted segmentation is effective when an upstream system can provide a target region.

This means the model is suitable for integration with detectors, trackers, grasp planners, or human-guided interfaces.

### 10.2 If automatic mask generation is slower or less reliable

The correct conclusion is:

> Automatic mask generation is useful for open-ended scene discovery, but it may be too slow or unstable for real-time robotic control without additional filtering.

### 10.3 If supervised baselines are faster

The correct conclusion is:

> When labeled data is available and real-time performance is required, supervised models may be more practical than large zero-shot foundation models.

### 10.4 If lightweight SAM variants reduce quality

The correct conclusion is:

> Lightweight variants are promising for edge deployment, but their use should be limited to scenarios where the observed quality-speed trade-off is acceptable.

### 10.5 If transparent, reflective, or small objects remain difficult

The correct conclusion is:

> Robotic deployment requires additional robustness mechanisms for difficult materials, small parts, occlusion, and dynamic scenes.

This congruence discipline is important for the final presentation. It shows that the project is critical and scientific rather than promotional.

---

## 11. Limitations

The project has several limitations.

First, much of the benchmark is simulation-based. Simulation enables controlled data generation, but it does not perfectly reproduce real sensor noise, lighting, material behavior, motion blur, and physical interaction. OCID helps address this limitation, but a larger real robot dataset would strengthen the conclusions.

Second, point and box prompt results are oracle-prompt evaluations. They isolate segmentation quality given a target cue, but they do not solve the problem of generating that cue in a fully autonomous system.

Third, the supervised baselines depend on labeled training data. Their performance should not be interpreted as zero-shot.

Fourth, speed measurements depend on hardware, implementation, image resolution, batch size, and prompt mode. Reported FPS values should therefore be interpreted as benchmark-specific rather than universal.

Fifth, the corrected common-test protocol is required for final comparative claims. Legacy full-dataset or validation-only metrics should not be used as final evidence.

Sixth, segmentation alone does not provide full scene understanding. A robot may also need object categories, affordances, pose estimates, depth, tracking, uncertainty, and symbolic task context.

---

## 12. Recommendations

The final recommendations should be conditional.

### 12.1 For high-quality prompt-guided segmentation

Use SAM ViT-H, SAM ViT-B, or SAM2 when segmentation quality is the main goal and when the system can provide a reliable point or box prompt. This setting is appropriate for offline analysis, interactive annotation, human-in-the-loop robotics, or high-level planning where latency is not the main constraint.

### 12.2 For prompt-guided robotic manipulation

Use box prompts when an upstream detector, tracker, grasp planner, or robot prior can provide a target region. Box prompts are cognitively meaningful because they represent task-driven attention: the robot is not segmenting everything, but focusing on a relevant object.

### 12.3 For real-time robotic control

Use supervised or lightweight models when the system requires high FPS. YOLOv8-seg and DeepLabV3+ are strong candidates when some labeled target-domain data is available. Lightweight SAM variants should be considered when the promptable interface is required but compute is limited.

### 12.4 For edge deployment

Use MobileSAM or EfficientSAM only after validating the relevant object categories and failure modes. The choice should depend on the required balance between quality, speed, checkpoint size, and CPU/GPU availability.

### 12.5 For transparent, reflective, occluded, and small objects

Do not rely on segmentation masks alone. Add robustness mechanisms such as temporal smoothing, depth cues, multi-view perception, uncertainty checking, task priors, or post-processing constraints.

---

## 13. Conclusion

This project evaluated foundation segmentation models as candidate perception modules for robotic scene understanding. The benchmark covers synthetic robot-centered scenes, a secondary synthetic dataset, and a real clutter dataset. It compares heavy foundation models, lightweight SAM variants, and supervised baselines using quality metrics, speed measurements, and failure-mode analysis.

The main conclusion is that promptable segmentation models are powerful but conditional tools for robotics. They can produce useful object masks in zero-shot settings, especially when a target prompt is available. However, robotic perception requires more than high average segmentation quality. A model must also be robust to materials, occlusion, small parts, motion, and real-time constraints.

From a COGAR perspective, segmentation is best understood as a cognitive interface: it transforms raw visual input into object-level regions that can support attention, scene representation, and action. Foundation segmentation models can contribute to this interface, but they must be integrated with prompt generation, temporal reasoning, domain validation, and task-aware decision-making.

The final recommendation is therefore not a single universal model. Instead, the correct model depends on the robotic scenario:

- Use SAM/SAM2 for high-quality prompt-guided segmentation.
- Use box prompts when a target cue is available from an upstream robotic module.
- Use supervised baselines for real-time deployment when labeled data exists.
- Use MobileSAM or EfficientSAM for edge-oriented trade-offs.
- Treat transparent, reflective, occluded, and small objects as high-risk cases requiring additional robustness.

---

## References

1. Kirillov, A., et al. **Segment Anything.** arXiv:2304.02643, 2023.  
   https://arxiv.org/abs/2304.02643

2. Ravi, N., et al. **SAM 2: Segment Anything in Images and Videos.** arXiv:2408.00714, 2024.  
   https://arxiv.org/abs/2408.00714

3. Zhao, X., et al. **Fast Segment Anything.** arXiv:2306.12156, 2023.  
   https://arxiv.org/abs/2306.12156

4. Zhang, C., et al. **MobileSAM: Faster Segment Anything.**  
   https://github.com/ChaoningZhang/MobileSAM

5. Xiong, Y., et al. **EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything.** arXiv:2312.00863, 2023.  
   https://arxiv.org/abs/2312.00863

6. He, K., Gkioxari, G., Dollár, P., and Girshick, R. **Mask R-CNN.** arXiv:1703.06870, 2017.  
   https://arxiv.org/abs/1703.06870

7. Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., and Adam, H. **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.** arXiv:1802.02611, 2018.  
   https://arxiv.org/abs/1802.02611

8. Ultralytics. **YOLOv8 Documentation.**  
   https://docs.ultralytics.com/models/yolov8/

9. University of Genoa. **Cognitive Architectures for Robotics course description.**  
   https://corsi.unige.it/en/off.f/2026/ins/93610

10. Project technical report. **Main Report: Foundation Model Segmentation Benchmark.**  
    `REPORT.md`

11. Project task documentation.  
    `docs/tasks/task1_dataset_creation.md`  
    `docs/tasks/task2_simulation_environment.md`  
    `docs/tasks/task3_robotic_platform.md`  
    `docs/tasks/task4_zero_shot_sam.md`  
    `docs/tasks/task5_classical_baselines.md`  
    `docs/tasks/task6_evaluation.md`  
    `docs/tasks/task7_inference_speed.md`  
    `docs/tasks/task8_failure_analysis.md`  
    `docs/tasks/task9_lightweight_sam.md`

---

## Appendix A: Suggested Figures to Include

Use the following figures from the repo if available:

```text
outputs/final_benchmark_assets/plots/dataset_examples.png
outputs/final_benchmark_assets/plots/zero_shot_miou_heatmap.png
outputs/final_benchmark_assets/plots/baseline_miou_bars.png
outputs/final_benchmark_assets/plots/cuda_speed_quality_scatter.png
outputs/final_benchmark_assets/plots/lightweight_sam_tradeoff_cuda.png
outputs/final_benchmark_assets/plots/challenge_group_weighted_iou.png
outputs/final_benchmark_assets/plots/zero_shot_dataset_prompt_winners.png
```

When using these plots, ensure that any plot derived from legacy summaries is labeled as legacy or regenerated after the corrected common-test evaluation is complete.

---

## Appendix B: Safe Presentation Wording

Use:

> The results suggest that foundation segmentation models can support robotic perception when prompts or object priors are available, but they require domain-specific validation before deployment.

Avoid:

> Foundation segmentation models solve robotic perception.

Use:

> Box prompts evaluate segmentation quality under an oracle target-region cue.

Avoid:

> Box-prompted models perform autonomous object discovery.

Use:

> Lightweight SAM variants provide deployment trade-offs.

Avoid:

> Lightweight SAM variants are always better for robotics.

Use:

> Supervised baselines remain important when labeled target-domain data and real-time performance are available.

Avoid:

> Zero-shot foundation models always outperform classical models.
