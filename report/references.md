# References and Source Map

Project: **Foundation Model Segmentation for Robotic Scenes**  
Assignment: **Zero-Shot Segmentation Benchmark for Robotic Perception (Simulation)**  
Student id: **5884715**

Recommended repo path:

```text
report/references.md
```

This file collects the main references used to frame the final report and presentation. It is organized as a source map rather than only a bibliography, so each reference also explains why it matters for this project.

---

## 1. Foundation Segmentation Models

### 1.1 Segment Anything Model, SAM

**Reference**  
Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., & Girshick, R. (2023). *Segment Anything*. arXiv:2304.02643.  
https://arxiv.org/abs/2304.02643

**Why it is used**  
This is the core foundation model behind the benchmark. SAM introduced promptable segmentation and was designed for zero-shot transfer to new image distributions and tasks.

**Use in this project**

- Supports the motivation for zero-shot segmentation.
- Supports the use of point, box, and automatic mask-generation prompting.
- Provides the baseline idea of a general-purpose segmentation module that may be reused in robotic perception without task-specific training.

**Suggested citation sentence**

> SAM introduced promptable segmentation as a general-purpose task and demonstrated that a model trained on large-scale mask data can transfer zero-shot to new segmentation settings.

---

### 1.2 Segment Anything Model 2, SAM2

**Reference**  
Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., Mintun, E., Pan, J., Alwala, K. V., Carion, N., Wu, C.-Y., Girshick, R., Dollár, P., & Feichtenhofer, C. (2024). *SAM 2: Segment Anything in Images and Videos*. arXiv:2408.00714.  
https://arxiv.org/abs/2408.00714

**Why it is used**  
SAM2 extends promptable segmentation from static images to images and videos. This is relevant for robotic scenes because robots perceive the world through continuous streams, not only isolated images.

**Use in this project**

- Supports the dynamic-scene and video-oriented motivation.
- Connects segmentation to temporal robotic perception.
- Strengthens the argument that foundation segmentation is moving toward interactive, streaming perception.

**Suggested citation sentence**

> SAM2 extends the segment-anything paradigm to images and videos using a transformer architecture with streaming memory, which is especially relevant for dynamic robotic perception.

---

### 1.3 FastSAM

**Reference**  
Zhao, X., Ding, W., An, Y., Du, Y., Yu, T., Li, M., Tang, M., & Wang, J. (2023). *Fast Segment Anything*. arXiv:2306.12156.  
https://arxiv.org/abs/2306.12156

**Why it is used**  
FastSAM addresses the runtime cost of SAM by reformulating the segment-anything task using an instance-segmentation-style approach followed by prompt-guided mask selection.

**Use in this project**

- Supports the speed-quality comparison among foundation segmentation models.
- Provides a faster zero-shot-style segmentation alternative.
- Helps evaluate whether a SAM-like approach can move closer to real-time robotic constraints.

**Suggested citation sentence**

> FastSAM attempts to reduce the computational burden of segment-anything-style segmentation by converting the task into segment generation followed by prompt-guided selection.

---

### 1.4 MobileSAM

**Reference**  
Zhang, C., Han, D., Qiao, Y., Kim, J. U., Bae, S.-H., Lee, S., & Hong, C. S. (2023). *Faster Segment Anything: Towards Lightweight SAM for Mobile Applications*. Project repository.  
https://github.com/ChaoningZhang/MobileSAM

**Why it is used**  
MobileSAM is a lightweight SAM variant designed for smaller deployment settings. It replaces the large SAM image encoder with a much smaller encoder while keeping the SAM-style prompt-and-mask-decoder pipeline.

**Use in this project**

- Supports the edge-deployment part of the benchmark.
- Helps evaluate whether a small SAM-like model can provide a useful speed-quality trade-off.
- Connects foundation segmentation to robotic platforms with limited compute.

**Suggested citation sentence**

> MobileSAM keeps the SAM-style prompting pipeline but replaces the heavyweight image encoder with a much smaller encoder, making it relevant for edge and mobile deployment.

---

### 1.5 EfficientSAM

**Reference**  
Xiong, Y., Varadarajan, B., Wu, L., Xiang, X., Xiao, F., Zhu, C., Dai, X., Wang, D., Sun, F., Iandola, F., Krishnamoorthi, R., & Chandra, V. (2023). *EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything*. arXiv:2312.00863.  
https://arxiv.org/abs/2312.00863

**Why it is used**  
EfficientSAM directly addresses the computational complexity of SAM by building lightweight SAM-style models through masked-image pretraining.

**Use in this project**

- Supports the lightweight-model comparison.
- Helps discuss the trade-off between segmentation quality, model size, and inference speed.
- Connects the benchmark to model-distillation and efficient foundation-model deployment.

**Suggested citation sentence**

> EfficientSAM proposes lightweight SAM-style models that reduce complexity while preserving useful segment-anything behavior.

---

## 2. Classical and Supervised Segmentation Baselines

### 2.1 Mask R-CNN

**Reference**  
He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). *Mask R-CNN*. arXiv:1703.06870.  
https://arxiv.org/abs/1703.06870

**Why it is used**  
Mask R-CNN is a standard instance-segmentation baseline. It detects object instances and predicts a mask for each instance.

**Use in this project**

- Provides a classical supervised instance-segmentation baseline.
- Helps compare zero-shot foundation models with models trained on a small labeled subset.
- Offers a familiar reference point for object-level robotic perception.

**Suggested citation sentence**

> Mask R-CNN is used as a classical supervised instance-segmentation baseline because it jointly detects objects and predicts instance masks.

---

### 2.2 DeepLabV3+

**Reference**  
Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation*. arXiv:1802.02611.  
https://arxiv.org/abs/1802.02611

**Why it is used**  
DeepLabV3+ is a strong semantic segmentation architecture that combines atrous spatial pyramid pooling with a decoder to recover sharper object boundaries.

**Use in this project**

- Provides a supervised semantic-segmentation baseline.
- Supports the boundary-quality comparison.
- Offers a contrast to object-instance models such as Mask R-CNN and YOLOv8-seg.

**Suggested citation sentence**

> DeepLabV3+ is included as a supervised semantic-segmentation baseline because its encoder-decoder design is intended to recover sharper object boundaries.

---

### 2.3 YOLOv8-seg / Ultralytics Segmentation

**Reference**  
Ultralytics. *Instance Segmentation with Ultralytics YOLO*. Documentation.  
https://docs.ultralytics.com/tasks/segment/

**Why it is used**  
YOLOv8-seg is a practical real-time instance-segmentation baseline. It is relevant for robotics because it can offer high throughput after task-specific training.

**Use in this project**

- Provides a speed-oriented supervised baseline.
- Supports real-time feasibility comparisons.
- Helps compare zero-shot generalization against small-subset fine-tuning.

**Suggested citation sentence**

> YOLOv8-seg is used as a practical supervised instance-segmentation baseline for comparing zero-shot foundation models with a real-time-oriented trained model.

---

## 3. Robotic Simulation and Synthetic Data Generation

### 3.1 Isaac Sim Synthetic Data Generation

**Reference**  
NVIDIA. *Synthetic Data Generation — Isaac Sim Documentation*.  
https://docs.isaacsim.omniverse.nvidia.com/6.0.0/synthetic_data_generation/index.html

**Why it is used**  
Isaac Sim supports synthetic data generation workflows for robotics, including perception data generation. It is suitable for robot-centered simulation scenes and ground-truth annotation.

**Use in this project**

- Supports the use of Isaac Sim for the Unitree G1 dataset.
- Provides justification for simulation-based annotation and controlled challenge generation.
- Connects the benchmark to robotics-oriented synthetic data generation.

**Suggested citation sentence**

> Isaac Sim provides robotics-oriented synthetic data generation tools, making it suitable for producing annotated robot-centered perception scenes.

---

### 3.2 BlenderProc

**Reference**  
Denninger, M., Sundermeyer, M., Winkelbauer, D., Zidan, Y., Olefir, D., Elbadrawy, M., Lodhi, A., & Katam, H. (2019). *BlenderProc*. arXiv:1911.01911.  
https://arxiv.org/abs/1911.01911

**Why it is used**  
BlenderProc is a procedural pipeline for generating realistic synthetic images and annotations, including segmentation, depth, normals, and pose information.

**Use in this project**

- Supports the secondary synthetic robotic-scene dataset.
- Provides a controlled environment for generating annotation-rich segmentation data.
- Helps diversify the simulation source beyond Isaac Sim.

**Suggested citation sentence**

> BlenderProc is used as a procedural synthetic-data pipeline because it can generate realistic images together with segmentation and other ground-truth annotations.

---

### 3.3 Synthetic Data and Sim-to-Real Motivation

**Reference**  
Greff, K., Belletti, F., Beyer, L., Doersch, C., Du, Y., Duckworth, D., Fleet, D. J., Golemo, F., Herrmann, C., Kipf, T., et al. (2022). *Kubric: A scalable dataset generator*. arXiv:2203.03570.  
https://arxiv.org/abs/2203.03570

**Why it is used**  
Kubric is not directly used in this project, but it is a useful background reference for the general motivation behind synthetic data: scalable generation, rich ground truth, control over scene conditions, and reduced annotation cost.

**Use in this project**

- Supports the broader argument for simulation-based dataset generation.
- Helps explain why synthetic data is attractive for segmentation benchmarks.
- Provides context for domain-gap discussion.

**Suggested citation sentence**

> Synthetic data generation is attractive for segmentation benchmarks because it provides scalable, controllable scenes with rich ground-truth annotations, although sim-to-real transfer remains a limitation.

---

## 4. Robotic Perception Datasets and Object Segmentation

### 4.1 OCID: Object Clutter Indoor Dataset

**Reference**  
Suchi, M., Patten, T., Fischinger, D., & Vincze, M. (2019). *EasyLabel: a semi-automatic pixel-wise object annotation tool for creating robotic RGB-D datasets*. OCID dataset page.  
https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-clutter-indoor-dataset/

**Why it is used**  
OCID provides pixel-wise annotated RGB-D clutter scenes for robot vision tasks such as object segmentation, classification, and recognition.

**Use in this project**

- Provides a real robotic clutter dataset for robustness and domain-gap evaluation.
- Complements the synthetic Isaac and BlenderProc datasets.
- Helps test whether simulation conclusions generalize toward real cluttered scenes.

**Suggested citation sentence**

> OCID is used as a real RGB-D clutter dataset to evaluate whether conclusions from synthetic robotic scenes remain plausible under real-world clutter conditions.

---

### 4.2 Synthetic Depth Data for Robotic Instance Segmentation

**Reference**  
Danielczuk, M., Matl, M., Gupta, S., Li, A., Lee, A., Mahler, J., & Goldberg, K. (2018). *Segmenting Unknown 3D Objects from Real Depth Images using Mask R-CNN Trained on Synthetic Data*. arXiv:1809.05825.  
https://arxiv.org/abs/1809.05825

**Why it is used**  
This paper is a useful robotics-specific reference showing how synthetic data can support object segmentation and grasping-oriented perception.

**Use in this project**

- Supports the robotics motivation for synthetic segmentation data.
- Connects segmentation to downstream grasping and object tracking.
- Strengthens the argument that synthetic robotic scenes are meaningful for perception benchmarking.

**Suggested citation sentence**

> Prior robotic perception work has shown that synthetic segmentation data can support unknown-object segmentation and downstream grasping pipelines.

---

## 5. Cognitive Architectures for Robotics and COGAR Connection

### 5.1 COGAR Course Page

**Reference**  
University of Genoa. *Cognitive Architectures for Robotics*. Course page.  
https://corsi.unige.it/en/off.f/2023/ins/66538

**Why it is used**  
The course page frames COGAR around cognition in humans and robots, software and cognitive architectures, robot design patterns, and practical robotics examples.

**Use in this project**

- Supports the cognitive-architecture framing of segmentation as a perception module.
- Connects the benchmark to embodiment, attention, perception-action loops, and robotic decision-making.
- Helps justify why the presentation should discuss more than technical scores.

**Suggested citation sentence**

> Within the COGAR framing, segmentation can be interpreted as a perceptual module that supports object attention, scene representation, and action selection inside a cognitive robotic architecture.

---

### 5.2 Cognitive Architectures and Robot Agency

**Reference**  
Vernon, D., Metta, G., & Sandini, G. (2007). *A survey of artificial cognitive systems: Implications for the autonomous development of mental capabilities in computational agents*. IEEE Transactions on Evolutionary Computation.  
https://doi.org/10.1109/TEVC.2007.902956

**Why it is used**  
This reference can be used for broader background on cognitive robotic systems and the need to integrate perception, action, learning, and representation.

**Use in this project**

- Supports the interpretation of segmentation as part of a larger cognitive system.
- Helps explain why perception quality alone is not sufficient: the robot must use perception for action.
- Provides theoretical grounding for the COGAR section.

**Suggested citation sentence**

> Cognitive robotic systems require the integration of perception, representation, learning, and action rather than isolated optimization of individual modules.

---

## 6. Metrics and Evaluation Concepts

### 6.1 COCO Evaluation and Mask AP

**Reference**  
Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). *Microsoft COCO: Common Objects in Context*. arXiv:1405.0312.  
https://arxiv.org/abs/1405.0312

**Why it is used**  
COCO is a standard benchmark framework for object detection and segmentation. Mask AP, AP50, and AP75 are common object-instance segmentation metrics.

**Use in this project**

- Supports the use of mask AP / AP50 / AP75.
- Provides standard evaluation context for instance segmentation.
- Helps justify using COCO-format annotations.

**Suggested citation sentence**

> Mask AP, AP50, and AP75 follow the standard object-instance segmentation evaluation style popularized by COCO.

---

### 6.2 Boundary F1

**Reference**  
Csurka, G., Larlus, D., Perronnin, F., & Meylan, F. (2013). *What is a good evaluation measure for semantic segmentation?* BMVC.  
https://www.bmva-archive.org.uk/bmvc/2013/Papers/paper0032/paper0032.pdf

**Why it is used**  
Boundary-aware metrics are useful because segmentation masks can have good area overlap while still being poor around object boundaries, which matters for robotic manipulation.

**Use in this project**

- Supports the use of boundary F1 alongside mIoU and mask AP.
- Helps explain why boundary quality matters for small parts and object edges.
- Connects segmentation evaluation to robotic manipulation reliability.

**Suggested citation sentence**

> Boundary-sensitive evaluation is important because area overlap alone may hide errors near object edges, which can be critical for robotic grasping and manipulation.

---

## 7. Project-Specific Sources

### 7.1 Project Repository

**Reference**  
Amirmat98. *COGAR Foundation-Model Segmentation for Robotic Scenes*. GitHub repository.  
https://github.com/amirmat98/COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes

**Why it is used**  
This is the project implementation repository containing scripts, configs, task documentation, benchmark outputs, plots, and reports.

**Use in this project**

- Main source for implementation details.
- Main source for dataset generation and evaluation workflow.
- Main source for plots, outputs, and reproducibility notes.

---

### 7.2 Technical Benchmark Report

**Reference**  
Amirmat98. *Main Report: Foundation Model Segmentation Benchmark*. `REPORT.md` in the project repository.

**Why it is used**  
This is the technical evidence tracker for the project. It records the closure status of Tasks 1-9, dataset summaries, public release information, artifact paths, evaluation notes, and final recommendation assets.

**Use in this project**

- Source for project-specific implementation status.
- Source for dataset sizes and artifact paths.
- Source for the warning that legacy validation/full-dataset metrics are archived and corrected common-test evaluation should be used for final comparisons.

---

## 8. Recommended In-Text Citation Placement

Use the references in the final report approximately like this:

| Report section | Most useful references |
|---|---|
| Research Problem | SAM, SAM2, OCID, COGAR course page |
| State of the Art | SAM, SAM2, FastSAM, MobileSAM, EfficientSAM, Mask R-CNN, DeepLabV3+, YOLOv8-seg |
| Research Formulation | Project `REPORT.md`, Task 4, Task 6, COCO |
| Cognitive Approach | COGAR course page, cognitive architectures background, SAM/SAM2 prompting |
| Dataset and Simulation | Isaac Sim documentation, BlenderProc, OCID, project `REPORT.md` |
| Benchmark Protocol | COCO, Mask R-CNN, DeepLabV3+, SAM/SAM2, project `REPORT.md` |
| Results | Project `REPORT.md`, corrected Task 6 outputs, final recommendation guide |
| Failure Analysis | Project `REPORT.md`, OCID, synthetic/robotic perception references |
| Limitations | Synthetic-data references, sim-to-real discussion, oracle-prompt explanation |
| Recommendations | Project outputs, MobileSAM, EfficientSAM, YOLOv8-seg, SAM/SAM2 |

---

## 9. Notes for Presentation Use

For slides, do not overload the audience with full citations. Use short labels:

- SAM, Kirillov et al., 2023
- SAM2, Ravi et al., 2024
- FastSAM, Zhao et al., 2023
- MobileSAM, Zhang et al., 2023
- EfficientSAM, Xiong et al., 2023
- Mask R-CNN, He et al., 2017
- DeepLabV3+, Chen et al., 2018
- Isaac Sim documentation, NVIDIA
- BlenderProc, Denninger et al., 2019
- OCID, TU Wien Vision for Robotics
- COGAR course page, University of Genoa

The full URLs can stay in the report and references file.
