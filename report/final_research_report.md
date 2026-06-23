# Final Research Report

**Project:** Foundation Model Segmentation for Robotic Scenes  
**Assignment:** Zero-Shot Segmentation Benchmark for Robotic Perception (Simulation)  
**Student id:** 5884715  

## Abstract

This project investigates whether promptable foundation segmentation models can be used as reliable perception modules for robotic scene understanding. The study focuses on challenging robotic conditions that are not fully represented by generic image benchmarks: reflective metal, transparent glass, partial occlusion, small screws and connectors, clutter, robot-body visibility, and dynamic objects.

The benchmark evaluates SAM ViT-H, SAM ViT-B, SAM2, FastSAM, MobileSAM, and EfficientSAM in zero-shot mode with point prompts, box prompts, and automatic mask generation where applicable. It compares them with supervised baselines trained on small labeled subsets: YOLOv8-seg, Mask R-CNN, and DeepLabV3+. The datasets include an Isaac Sim Unitree G1 synthetic dataset, a BlenderProc COGAR-SimRobotics dataset, and OCID as a real-world clutter reference.

The main conclusion is conditional. Foundation segmentation models are useful for robotic perception, especially when a robot or an upstream module can provide reliable prompts. However, they are not complete robotic perception systems by themselves. Robotic deployment still requires prompt generation, temporal consistency, domain validation, real-time analysis, and explicit handling of failure cases.

---

## 1. Research Problem

Robotic perception requires object-level scene understanding before a robot can manipulate, inspect, avoid, track, or reason about objects. Segmentation masks provide this object-level interface: they define where an object is, how it is shaped, where its boundary lies, and which pixels may be relevant for action.

The research problem addressed in this project is:

> Can promptable foundation segmentation models provide reliable zero-shot object masks for robotic perception in challenging simulated scenes, and how do they compare with lightweight and supervised alternatives in terms of accuracy, robustness, and real-time feasibility?

This problem has three dimensions.

First, there is an accuracy problem. The model must produce masks that overlap well with ground truth and preserve object boundaries. This is measured using mIoU, boundary F1, mask AP, AP50, AP75, and per-category IoU.

Second, there is a robustness problem. Robotic scenes contain materials and configurations that are difficult for visual segmentation: glass, shiny metal, partial occlusions, thin cables, small screws, connectors, robot limbs, cluttered work surfaces, and moving objects. A model with high average performance may still fail on objects that matter for manipulation or safety.

Third, there is a deployment problem. A model may be accurate but too slow for closed-loop control. Robotic systems often need perception results under real-time or near-real-time constraints, especially when segmentation feeds tracking, grasping, navigation, or reactive planning.

Therefore, the project does not ask only “which model has the best score?” It asks which model is appropriate for each robotic scenario:

- high-quality prompt-guided segmentation,
- automatic object discovery,
- small labeled-data supervised deployment,
- lightweight or edge deployment,
- CPU/GPU real-time feasibility,
- robustness under difficult robotic scene conditions.

---

## 2. State of the Art

### 2.1 Promptable foundation segmentation

Segment Anything Model (SAM) introduced a general promptable segmentation paradigm. Instead of training a separate segmentation model for each task, SAM accepts prompts such as points and boxes and predicts masks. This makes it attractive for robotics because robots often encounter new objects and environments where dense labels are expensive.

SAM2 extends the segment-anything idea to images and videos using a streaming-memory design. This is important for robotics because robots perceive continuous scenes, not isolated static images. Temporal consistency is relevant for tracking moving objects, maintaining object identity through occlusion, and supporting perception-action loops.

### 2.2 Efficient and lightweight variants

The main limitation of large SAM-style models is computational cost. SAM ViT-H and similar large models can provide strong masks but may be too slow for real-time robotic control.

FastSAM, MobileSAM, and EfficientSAM address this issue from different directions. FastSAM uses a faster segment-anything-style pipeline. MobileSAM keeps a SAM-compatible interface but replaces the heavy image encoder with a smaller one. EfficientSAM uses efficient masked-image pretraining to produce smaller SAM-style models. These variants are relevant for robots with limited GPU memory, mobile platforms, embedded systems, and edge deployment.

### 2.3 Classical supervised baselines

Classical supervised segmentation remains important. Mask R-CNN is a standard instance segmentation method. DeepLabV3+ is a strong semantic segmentation model with decoder-based boundary refinement. YOLOv8-seg is a practical real-time instance segmentation baseline.

These baselines answer a different question from zero-shot SAM models. SAM-family models test whether segmentation can generalize without task-specific training. Supervised baselines test what can be achieved when a small amount of target-domain labeled data is available. The comparison is useful for robotics because real systems often combine pretrained models with limited task-specific adaptation.

### 2.4 Research gap

The state of the art in general segmentation is strong, but robotic perception is embodied, situated, and action-oriented. A mask error in a robotic system is not only a visual error. It can affect grasp planning, collision checking, object tracking, navigation, or human-robot interaction.

Existing segmentation models are not automatically reliable in robotic scenes because:

- transparent objects may have weak or misleading boundaries,
- reflective surfaces can produce false visual edges,
- small parts occupy few pixels,
- occlusions create ambiguous object extents,
- robot-body parts may merge visually with tools or workbench objects,
- automatic mask generation may produce duplicate or merged proposals,
- large foundation models may be too slow for robot control loops.

This project is positioned in that gap. It evaluates segmentation models as candidate perception components for robotic cognitive architectures, not only as generic computer-vision models.

---

## 3. Research Formulation

### 3.1 Objectives

The objectives are:

1. Create and curate robotic-scene datasets with segmentation annotations.
2. Include robotics-specific challenges: reflective metal, transparent glass, partial occlusion, small parts, and moving/dynamic objects.
3. Use simulation environments, especially Isaac Sim and BlenderProc, to generate controlled scenes.
4. Use a simulated robotic platform, especially Unitree G1, so the main dataset is robot-centered.
5. Run SAM ViT-H, SAM ViT-B, SAM2, FastSAM, MobileSAM, and EfficientSAM in zero-shot mode.
6. Evaluate point prompts, box prompts, and automatic mask generation.
7. Train YOLOv8-seg, Mask R-CNN, and DeepLabV3+ on small labeled subsets for supervised comparison.
8. Measure mIoU, boundary F1, mask AP, AP50, AP75, per-category IoU, and challenge-group behavior.
9. Measure GPU and CPU inference speed.
10. Analyze qualitative failure modes and produce model-selection recommendations.

### 3.2 Hypotheses

The benchmark is guided by five hypotheses.

**H1: Prompted segmentation will be more reliable than automatic mask generation.**  
Point and box prompts provide a target cue. Automatic mode must discover objects without a target and is expected to be more unstable in clutter.

**H2: Box prompts will generally outperform point prompts.**  
A box prompt contains stronger spatial information than a single point and should reduce ambiguity in cluttered robotic scenes.

**H3: Large SAM-family models will provide strong mask quality but weak real-time feasibility.**  
Large foundation models are expected to perform well in quality metrics, but their runtime may be unsuitable for closed-loop robotic control.

**H4: Lightweight SAM variants will improve deployability but may reduce robustness.**  
MobileSAM and EfficientSAM are expected to provide better speed-size trade-offs, but may struggle with boundaries, clutter, small parts, and difficult materials.

**H5: Supervised baselines can be more practical when labeled target-domain data is available.**  
YOLOv8-seg, Mask R-CNN, and DeepLabV3+ are not zero-shot, but they can be faster or more consistent after small-subset fine-tuning.

### 3.3 Methodology

The project follows a benchmark methodology:

1. Generate or curate datasets.
2. Normalize annotations into COCO-compatible segmentation format.
3. Create deterministic train/validation/test splits.
4. Generate point and box prompts from ground-truth masks and boxes for oracle-prompt evaluation.
5. Run zero-shot SAM-family models without dataset-specific training.
6. Train supervised baselines on small labeled subsets.
7. Evaluate all model families with the same metric implementation.
8. Measure inference speed on GPU and CPU.
9. Group per-category results into robotic challenge groups.
10. Inspect representative visual failures.

The datasets are:

| Dataset | Type | Role |
| --- | --- | --- |
| Isaac official Unitree G1 | Synthetic, Isaac Sim | Main robot-centered benchmark |
| BlenderProc COGAR-SimRobotics-1000 | Synthetic, BlenderProc | Controlled synthetic challenge benchmark |
| OCID | Real clutter dataset | Domain-gap and real-world clutter reference |

The prompt protocol is important. Point and box prompts are derived from ground-truth annotations. This does not represent a fully autonomous system; it isolates the segmentation model’s ability once a target cue is available. In a robot, that cue would need to come from a detector, tracker, planner, human operator, or task prior.

The corrected comparison protocol uses common held-out test splits for final conclusions. Validation data is used for checkpoint selection, not final comparison. This avoids unfair comparison between zero-shot models evaluated on full datasets and supervised baselines evaluated on validation subsets.

### 3.4 Metrics

The evaluation uses:

- **mIoU:** region overlap quality,
- **boundary F1:** contour accuracy, important for manipulation and grasping,
- **mask AP/AP50/AP75:** instance-level segmentation quality,
- **per-category IoU:** object-type-specific performance,
- **challenge-group IoU:** robotic condition robustness,
- **FPS and latency:** real-time feasibility,
- **qualitative overlays:** interpretation of representative failures.

---

## 4. Cognitive Approach

This project is connected to Cognitive Architectures for Robotics because segmentation is treated as a perception module inside an embodied cognitive system, not as an isolated image-processing output.

A cognitive robotic architecture must transform raw sensory input into internal representations that can support attention, memory, reasoning, planning, and action. Segmentation masks are one possible intermediate representation. They convert pixels into object-level regions that can be used for:

- selecting a task-relevant object,
- tracking object state,
- estimating graspable regions,
- separating object from background,
- reasoning about occlusion,
- checking collision or free space,
- linking perception to symbolic or task-level representations.

The COGAR connection can be summarized as follows:

| COGAR concept | Connection to this project |
| --- | --- |
| Embodiment | The main dataset uses robot-centered Unitree G1 simulation rather than only generic images. |
| Situated perception | Models are evaluated in clutter, occlusion, robot-body visibility, material effects, and dynamic scenes. |
| Attention | Point and box prompts represent task-driven attention cues. |
| Scene representation | Masks provide object-level regions for downstream reasoning. |
| Perception-action loop | Mask errors can affect grasping, tracking, navigation, and planning. |
| Real-time cognition | FPS and latency determine whether the model can be used inside a control loop. |
| Robustness | Failure analysis identifies conditions where perception becomes unsafe or unreliable. |

Point prompts can be interpreted as minimal attentional signals. Box prompts can be interpreted as stronger top-down task priors. Automatic mask generation represents open-ended scene parsing. This distinction is cognitively important: a robot rarely perceives passively. It usually perceives in relation to a goal, action, or uncertainty.

From this perspective, the benchmark evaluates whether foundation segmentation can serve as an object-perception layer in a cognitive architecture. The answer is conditional: it can be useful, but only if integrated with prompt generation, temporal reasoning, domain checks, and action-aware validation.

---

## 5. Congruence of Results and Conclusions

This section checks that the conclusions follow from the benchmark evidence and do not overclaim beyond the results.

### 5.1 Main quantitative patterns

The aggregate benchmark results show that box-prompted large SAM models provide the strongest quality in the synthetic datasets. For example, the benchmark assets identify SAM ViT-H with box prompts as the best CUDA-quality result on BlenderProc and Isaac, with approximately 0.923 mIoU on BlenderProc and 0.752 mIoU on Isaac. This supports the hypothesis that strong spatial prompts improve segmentation quality.

The same results also show that high-quality SAM models are not real-time in this benchmark. SAM ViT-H box prompting is below 1 FPS on CUDA in the recorded speed tests. Therefore, the correct conclusion is not “use SAM ViT-H for all robotics.” The congruent conclusion is:

> Use large SAM models when mask quality is the priority and latency is acceptable, or when the system is offline, human-guided, or high-level planning oriented.

For speed-quality trade-off, the benchmark assets identify supervised baselines as strong practical options. YOLOv8-seg reaches about 41.6 FPS on BlenderProc with high mIoU, while DeepLabV3+ reaches about 34-38 FPS on Isaac/OCID in the recorded CUDA tests. This supports the conclusion that supervised models remain important when real-time performance and some labeled target-domain data are available.

Lightweight SAM variants provide a middle ground. MobileSAM with box prompts reaches about 15-17 FPS in the recorded CUDA tests while keeping useful promptable segmentation quality. This supports the conclusion that lightweight SAM variants are relevant for edge-oriented promptable perception, but not necessarily superior in accuracy to large SAM models.

Automatic mask generation is weaker for real-time robotics. It is generally slower and more prone to duplicate, merged, or class-agnostic masks. The congruent conclusion is:

> Automatic mode is useful for open-ended proposal discovery, but it should be filtered or constrained before use in closed-loop robotic behavior.

### 5.2 Failure-mode congruence

The failure analysis identifies the most important robotic risks:

- small parts and thin structures,
- transparent and reflective surfaces,
- robot-body and occlusion cases,
- cluttered support surfaces,
- dynamic or moving objects.

These failures are consistent with the research problem. They show that average mask quality is not enough for robotic deployment. A robot may fail if it misses a screw, merges a glass object with the background, segments a reflection instead of an object, or confuses a robot part with a tool.

Therefore, the final conclusion must not say that foundation segmentation solves robotic perception. The evidence supports a narrower and more defensible claim:

> Foundation segmentation models can support robotic perception, especially under prompt-guided conditions, but robotic deployment requires additional robustness mechanisms for materials, occlusion, small objects, motion, and real-time constraints.

### 5.3 Model-selection recommendations

The conclusions are scenario-dependent.

| Robotic scenario | Recommended model family | Reason |
| --- | --- | --- |
| Highest quality, offline or high-level planning | SAM ViT-H / SAM2 with box prompts | Strong mask quality when latency is acceptable |
| Prompt-guided manipulation | SAM/SAM2/MobileSAM with box prompts | Box prompts represent task-driven attention and reduce ambiguity |
| Real-time control with labeled data | YOLOv8-seg or DeepLabV3+ | Higher FPS after target-domain supervision |
| Edge-oriented promptable perception | MobileSAM or EfficientSAM | Better speed-size trade-off than heavy SAM |
| Open-ended object discovery | Automatic mask generation, followed by filtering | Useful proposals but not directly reliable for control |
| Transparent/reflective/small/occluded objects | Any model plus extra validation | High-risk categories need depth, temporal, task, or uncertainty checks |

### 5.4 Limits of the conclusions

The conclusions are limited by the benchmark design.

First, much of the dataset is synthetic. Simulation provides controlled annotations and challenge coverage, but it cannot perfectly reproduce real sensor noise, lighting, motion blur, or physical material behavior.

Second, point and box prompts are oracle prompts derived from ground truth. They evaluate segmentation quality given a target cue, not the full autonomy of a robot that must generate the cue itself.

Third, supervised baselines are not zero-shot. Their performance depends on labeled training data.

Fourth, speed results depend on hardware, implementation, resolution, model checkpoint, prompt mode, and preprocessing/post-processing choices.

Fifth, segmentation alone is not complete scene understanding. A robot may also need object categories, poses, depth, affordances, uncertainty estimates, tracking, and symbolic task context.

### 5.5 Final conclusion

The final conclusion is:

> Promptable foundation segmentation models are valuable perception modules for robotic scene understanding, but they are conditional tools. They are most useful when integrated into a broader cognitive robotic architecture that supplies prompts, validates masks, reasons over time, and selects models according to task constraints.

This conclusion is congruent with the results. The benchmark supports the use of SAM-family models for high-quality prompt-guided segmentation, supports lightweight variants for deployment trade-offs, supports supervised baselines for real-time use when labeled data exists, and identifies transparent, reflective, occluded, small, and dynamic objects as high-risk failure cases.

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

8. Ultralytics. **Instance Segmentation with Ultralytics YOLO.**  
   https://docs.ultralytics.com/tasks/segment/

9. Project task documentation and benchmark artifacts in this repository: `docs/tasks/`, `REPORT.md`, and `outputs/final_benchmark_assets/`.
