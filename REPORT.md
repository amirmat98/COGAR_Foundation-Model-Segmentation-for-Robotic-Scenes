# Foundation Model Segmentation for Robotic Scenes

**Assignment:** Zero-Shot Segmentation Benchmark for Robotic Perception  
**Student id:** 5884715  

## Abstract

This project investigates whether promptable foundation segmentation models can
serve as reliable perception modules for robotic scene understanding. The
benchmark focuses on visual conditions that are common in robotics but difficult
for generic image segmentation: reflective metal, transparent glass, partial
occlusion, small screws and connectors, clutter, robot-body visibility, and
dynamic objects.

The study evaluates SAM ViT-H, SAM ViT-B, SAM2, FastSAM, MobileSAM, and
EfficientSAM in zero-shot settings with point prompts, box prompts, and
automatic mask generation where applicable. It compares them with supervised
baselines trained on small labeled subsets: YOLOv8-seg, Mask R-CNN, and
DeepLabV3+. The datasets include an Isaac Sim Unitree G1 synthetic dataset, a
BlenderProc COGAR-SimRobotics synthetic dataset, and OCID as a real clutter
reference.

The conclusion is conditional. Foundation segmentation models are useful for
robotic perception, especially when a robot or upstream module can provide a
reliable target prompt. They are not complete robotic perception systems by
themselves. Deployment still requires prompt generation, temporal consistency,
domain validation, inference-speed analysis, and explicit handling of failure
cases.

---

## 1. Research Problem

Robots need object-level scene understanding before they can manipulate,
inspect, avoid, track, or reason about objects. A segmentation mask is therefore
not only an image output. It can become an object-level representation used by a
grasp planner, tracker, navigation module, inspection system, or higher-level
cognitive architecture.

The research problem is:

> Can promptable foundation segmentation models provide reliable zero-shot
> object masks for robotic perception in challenging simulated scenes, and how
> do they compare with lightweight and supervised alternatives in terms of
> accuracy, robustness, and real-time feasibility?

This problem has three parts.

First, the model must produce accurate masks. Region overlap, boundary quality,
and instance-level precision matter because downstream robotic modules depend on
object shape and location.

Second, the model must be robust under robotic scene conditions. Transparent
objects, reflective objects, small parts, occlusions, clutter, dynamic objects,
and robot-body regions can all break assumptions learned from generic images.

Third, the model must be deployable. A high-quality model may still be
unsuitable for closed-loop control if it is too slow or too large for the target
robot hardware.

The project therefore does not search for one universal winner. It asks which
segmentation model and prompting strategy is suitable for each robotic use case:
high-quality offline perception, prompt-guided manipulation, automatic object
proposal generation, real-time control, or edge deployment.

---

## 2. State of the Art

### 2.1 Promptable foundation segmentation

Segment Anything Model (SAM) introduced a general promptable segmentation
paradigm. Instead of training a separate segmentation model for every task, SAM
accepts prompts such as points and boxes and predicts masks. This makes SAM
attractive for robotics because robots often encounter new objects and
environments where dense labels are expensive.

SAM2 extends the segment-anything idea to images and videos through a
streaming-memory design. This is relevant for robotics because robot perception
is normally continuous. Dynamic objects, temporary occlusions, and object
tracking require temporal consistency rather than only single-frame masks.

### 2.2 Lightweight SAM variants

Large SAM-style models can be computationally expensive. This is a problem for
mobile robots, embedded platforms, and real-time systems. FastSAM, MobileSAM,
and EfficientSAM address this limitation in different ways.

FastSAM reformulates segment-anything-style segmentation with a faster
instance-segmentation pipeline. MobileSAM keeps the SAM-style prompt interface
but replaces the heavy encoder with a smaller one. EfficientSAM uses efficient
masked-image pretraining to build compact SAM-style models. These variants are
important because robotic deployment is constrained by latency, memory, power,
and hardware availability.

### 2.3 Classical supervised baselines

Classical supervised segmentation remains relevant. Mask R-CNN is a standard
instance segmentation architecture. DeepLabV3+ is a semantic segmentation model
designed for strong dense prediction and boundary recovery. YOLOv8-seg is a
real-time instance segmentation baseline.

These baselines answer a different question from zero-shot SAM models. SAM-style
models test generalization without task-specific training. Supervised baselines
test what can be achieved when a small amount of target-domain labeled data is
available. Robotics often needs this comparison because a practical system may
combine pretrained foundation models with limited domain adaptation.

### 2.4 Robotic perception gap

The state of the art in general segmentation is strong, but robotic perception
is embodied, situated, and action-oriented. A segmentation error in a robot is
not only a visual error. It can affect grasping, collision checking, object
tracking, navigation, or human-robot interaction.

General image segmentation benchmarks do not fully test:

- reflective and transparent materials,
- robot-centered viewpoints,
- small mechanical parts,
- occlusion by the robot or other objects,
- dynamic scenes,
- real-time latency constraints,
- integration with attention and action.

This project addresses that gap by evaluating segmentation models as candidate
perception components inside a robotic cognitive architecture.

---

## 3. Research Formulation

### 3.1 Objectives

The objectives are:

1. Create and curate robotic-scene datasets with segmentation annotations.
2. Include reflective metal, transparent glass, partial occlusion, small parts,
   and moving/dynamic objects.
3. Use simulation environments such as Isaac Sim and BlenderProc.
4. Use a simulated robotic platform, especially Unitree G1, to make the main
   dataset robot-centered.
5. Run SAM ViT-H, SAM ViT-B, SAM2, FastSAM, MobileSAM, and EfficientSAM in
   zero-shot mode.
6. Evaluate point prompts, box prompts, and automatic mask generation.
7. Train YOLOv8-seg, Mask R-CNN, and DeepLabV3+ on small labeled subsets.
8. Measure mIoU, boundary F1, mask AP, AP50, AP75, per-category IoU, and
   challenge-group behavior.
9. Measure GPU and CPU inference speed.
10. Analyze qualitative failure modes and produce model-selection
    recommendations.

### 3.2 Hypotheses

| Hypothesis | Expected behavior |
| --- | --- |
| H1 | Prompted segmentation should be more reliable than automatic mask generation because a target cue reduces ambiguity. |
| H2 | Box prompts should generally outperform point prompts because they provide stronger spatial constraints. |
| H3 | Large SAM-family models should provide strong quality but weak real-time feasibility. |
| H4 | Lightweight SAM variants should improve deployment feasibility but may lose quality on difficult boundaries and clutter. |
| H5 | Supervised baselines can be more practical for real-time robotic loops when labeled target-domain data is available. |
| H6 | Transparent, reflective, occluded, small, and dynamic objects should remain high-risk failure cases. |

### 3.3 Dataset design

The benchmark uses two generated synthetic datasets and one real clutter
reference dataset.

| Dataset | Type | Role | Images |
| --- | --- | --- | ---: |
| Isaac official Unitree G1 | Synthetic, Isaac Sim | Main robot-centered simulation benchmark | 1000 |
| BlenderProc COGAR-SimRobotics-1000 | Synthetic, BlenderProc | Controlled synthetic challenge benchmark | 1000 |
| OCID | Real RGB-D clutter dataset | Real-world clutter and domain-gap reference | 2390 |

![Dataset examples](outputs/final_benchmark_assets/plots/dataset_examples.png)

The Isaac dataset uses the official Unitree G1 asset and contains robot-centered
scenes. This is important because the benchmark is not only about generic
object segmentation. It tests segmentation in the visual context of a robot
workspace.

### 3.4 Model and prompt protocol

| Group | Models | Evaluation mode |
| --- | --- | --- |
| Heavy zero-shot models | SAM ViT-H, SAM ViT-B, SAM2, FastSAM | Point, box, automatic |
| Lightweight SAM variants | MobileSAM, EfficientSAM-Ti, EfficientSAM-S | Point, box, automatic/grid automatic |
| Supervised baselines | YOLOv8-seg, Mask R-CNN, DeepLabV3+ | Inference after small-subset training |

Point and box prompt results are oracle-prompt evaluations. The prompts are
derived from ground-truth masks and boxes to isolate segmentation quality once a
target cue is available. In a deployed robot, such prompts would need to come
from an upstream detector, tracker, planner, human operator, or task prior.

Automatic mask generation is closer to prompt-free object discovery. It is more
autonomous but less controlled and often slower.

### 3.5 Metrics

The evaluation uses:

- **mIoU** for region overlap,
- **boundary F1** for contour quality,
- **mask AP / AP50 / AP75** for instance-level segmentation,
- **per-category IoU** for object-specific performance,
- **challenge-group IoU** for robotic robustness,
- **FPS and latency** for deployment feasibility,
- **qualitative overlays** for failure interpretation.

Final comparative evaluation uses a common held-out test protocol so that
zero-shot models and supervised baselines are compared on the same reserved
image IDs.

---

## 4. Cognitive Approach

This project is connected to Cognitive Architectures for Robotics because
segmentation is treated as a perception module inside an embodied cognitive
system, not as an isolated image-processing output.

A cognitive robotic architecture must transform raw sensory input into
representations that support attention, memory, reasoning, planning, and
action. Segmentation masks are one such intermediate representation. They
convert pixels into object-level regions that can be used for:

- selecting a task-relevant object,
- tracking object state,
- estimating graspable regions,
- separating object from background,
- reasoning about occlusion,
- checking collision or free space,
- linking perception to symbolic or task-level representations.

| COGAR concept | Connection to this project |
| --- | --- |
| Embodiment | The main dataset uses robot-centered Unitree G1 simulation. |
| Situated perception | Models are tested in clutter, occlusion, material effects, robot-body visibility, and dynamic scenes. |
| Attention | Point and box prompts represent task-driven attention cues. |
| Scene representation | Masks provide object-level regions for downstream reasoning. |
| Perception-action loop | Mask errors can affect grasping, tracking, navigation, and planning. |
| Real-time cognition | FPS and latency determine whether segmentation can run inside a control loop. |
| Robustness | Failure analysis identifies conditions where perception becomes unreliable for action. |

Point prompts can be interpreted as minimal attentional signals. Box prompts are
stronger top-down priors. Automatic mask generation represents bottom-up scene
parsing. This distinction matters because robotic perception is usually
goal-directed: the robot perceives in relation to a task, an action, or an
uncertainty that must be resolved.

---

## 5. Congruence of Results and Conclusions

This section connects the obtained results to the final conclusions. The goal is
to avoid overclaiming. The results support conditional recommendations rather
than a universal statement that one model is always best.

### 5.1 Segmentation quality

The aggregate benchmark results show that box-prompted large SAM models provide
the strongest quality on the synthetic datasets. In the final benchmark asset
tables, SAM ViT-H with box prompts is the best CUDA-quality model on
BlenderProc and Isaac.

| Dataset | Best quality model | Prompt | mIoU | Boundary F1 | Mask AP | CUDA FPS |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| BlenderProc | SAM ViT-H | Box | 0.923 | 0.905 | 0.868 | 0.574 |
| Isaac G1 | SAM ViT-H | Box | 0.752 | 0.874 | 0.678 | 0.574 |
| OCID | DeepLabV3+ | Inference | 0.963 | 0.880 | N/A | 37.811 |

![Zero-shot mIoU heatmap](outputs/final_benchmark_assets/plots/zero_shot_miou_heatmap.png)

This supports H1 and H2: prompt cues matter, and box prompts are especially
effective because they provide stronger spatial information than a single
point.

The conclusion must remain precise. Strong box-prompt results do not mean that
the model performs fully autonomous object discovery. They mean that the model
segments well when a target region is supplied by an oracle or upstream robotic
module.

### 5.2 Supervised baselines and real-time feasibility

Large SAM models provide strong quality but are slow in the recorded speed
tests. SAM ViT-H with box prompts is below 1 FPS on CUDA. This makes it more
suitable for offline analysis, high-level planning, or human-guided operation
than for fast closed-loop control.

The best speed-quality trade-off results show why supervised baselines remain
important.

| Dataset | Best CUDA trade-off | mIoU | CUDA FPS | Interpretation |
| --- | --- | ---: | ---: | --- |
| BlenderProc | YOLOv8-seg | 0.861 | 41.647 | Strong real-time instance segmentation after supervision |
| Isaac G1 | DeepLabV3+ | 0.660 | 34.332 | Real-time semantic segmentation on the robot-centered dataset |
| OCID | DeepLabV3+ | 0.963 | 37.811 | Strong real-time semantic segmentation on clutter scenes |

![Classical baseline mIoU bars](outputs/final_benchmark_assets/plots/baseline_miou_bars.png)

![CUDA speed-quality trade-off](outputs/final_benchmark_assets/plots/cuda_speed_quality_scatter.png)

This supports H3 and H5. Large foundation models are strong but expensive.
Supervised baselines can be more practical when labeled target-domain data is
available and the robot needs high FPS.

### 5.3 Lightweight SAM variants

Lightweight SAM variants provide a middle ground between heavy promptable
foundation models and supervised real-time baselines.

| Dataset | Best lightweight prompted model | Prompt | mIoU | CUDA FPS | Checkpoint size |
| --- | --- | --- | ---: | ---: | ---: |
| BlenderProc | MobileSAM | Box | 0.883 | 15.737 | 40.73 MB |
| Isaac G1 | MobileSAM | Box | 0.693 | 16.895 | 40.73 MB |
| OCID | MobileSAM | Box | 0.824 | 15.522 | 40.73 MB |

![Lightweight SAM trade-off](outputs/final_benchmark_assets/plots/lightweight_sam_tradeoff_cuda.png)

This supports H4. MobileSAM and EfficientSAM are useful for edge-oriented
promptable perception, but they should be selected by explicit speed-quality
trade-off rather than assumed to preserve the full capability of larger SAM
models.

### 5.4 Robotic challenge groups and failure modes

Average metrics do not fully describe robotic reliability. A robot may fail if
it misses a small connector, merges a transparent object with the background,
segments a reflection instead of the object, or confuses a robot part with an
external tool.

The failure analysis identifies the highest-risk groups:

| Challenge group | Why it matters for robotics |
| --- | --- |
| Small parts and thin structures | Screws, cables, and connectors are important for assembly and inspection but occupy few pixels. |
| Transparent and reflective surfaces | Glass and metal can weaken or distort visual boundaries. |
| Robot body and occlusion | The robot can occlude the scene or visually merge with task objects. |
| Dynamic objects | Motion can break single-frame segmentation and tracking assumptions. |
| Cluttered support surfaces | Workbenches and bins increase ambiguity and object merging. |

![Robotic challenge group performance](outputs/final_benchmark_assets/plots/challenge_group_weighted_iou.png)

This supports H6. Robotic deployment requires additional validation for
materials, occlusion, small objects, and motion.

### 5.5 Prompt-mode interpretation

The benchmark also shows why prompt mode must be interpreted carefully.

| Prompt mode | Robotic interpretation | Main conclusion |
| --- | --- | --- |
| Point | Minimal attention cue | Useful but ambiguous in clutter |
| Box | Strong task or detector prior | Best quality when target region is available |
| Automatic | Bottom-up proposal generation | More autonomous but slower and less controlled |

![Best zero-shot model by dataset and prompt](outputs/final_benchmark_assets/plots/zero_shot_dataset_prompt_winners.png)

The strongest conclusion is not that foundation models replace robotic
perception. The supported conclusion is that they can become useful perception
modules when integrated with prompt generation, validation, and task context.

### 5.6 Final recommendations

| Robotic scenario | Recommended model family | Reason |
| --- | --- | --- |
| Highest mask quality, offline or high-level planning | SAM ViT-H / SAM2 with box prompts | Strong mask quality when latency is acceptable |
| Prompt-guided manipulation | SAM/SAM2/MobileSAM with box prompts | Box prompts represent task-driven attention and reduce ambiguity |
| Real-time control with labeled data | YOLOv8-seg or DeepLabV3+ | Higher FPS after target-domain supervision |
| Edge-oriented promptable perception | MobileSAM or EfficientSAM | Better speed-size trade-off than heavy SAM |
| Open-ended object discovery | Automatic mask generation plus filtering | Useful proposals but not directly reliable for control |
| Transparent, reflective, occluded, or small objects | Any model plus extra validation | High-risk categories need depth, temporal checks, task priors, or uncertainty handling |

### 5.7 Limitations

The conclusions are limited by the benchmark design.

First, much of the dataset is synthetic. Simulation provides controlled
annotations and challenge coverage, but it cannot perfectly reproduce real
sensor noise, motion blur, lighting, and physical material behavior.

Second, point and box prompt results are oracle-prompt evaluations. They
evaluate segmentation quality given a target cue, not the full autonomy of a
robot that must generate the cue itself.

Third, supervised baselines are not zero-shot. Their performance depends on
labeled target-domain data.

Fourth, speed results depend on hardware, implementation, image resolution,
checkpoint, prompt mode, preprocessing, and post-processing.

Fifth, segmentation alone is not complete scene understanding. A robot may also
need object categories, poses, depth, affordances, uncertainty estimates,
tracking, and symbolic task context.

### 5.8 Final conclusion

Promptable foundation segmentation models are valuable perception modules for
robotic scene understanding, but they are conditional tools. They are most
useful when integrated into a broader cognitive robotic architecture that
supplies prompts, validates masks, reasons over time, and selects models
according to task constraints.

The benchmark supports:

- SAM-family models for high-quality prompt-guided segmentation,
- lightweight SAM variants for edge-oriented trade-offs,
- supervised baselines for real-time deployment when labeled data exists,
- explicit failure-mode analysis for transparent, reflective, occluded, small,
  and dynamic objects.

The final recommendation is therefore task-dependent rather than model-universal.
Robotic perception should choose the segmentation model according to quality,
prompt availability, robustness, runtime, and downstream action requirements.

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

9. Project technical artifacts and task documentation: [README.md](README.md),
   [docs/tasks/](docs/tasks), [report/](report), and
   [outputs/final_benchmark_assets/](outputs/final_benchmark_assets).
