# State of the Art

Project: **Foundation Model Segmentation for Robotic Scenes**  
Assignment: **Zero-Shot Segmentation Benchmark for Robotic Perception (Simulation)**  
Student id: **5884715**

Recommended repo path:

```text
presentation/02_state_of_the_art.md
```

This file provides the state-of-the-art section for the final presentation and written report. It explains the research context behind promptable foundation segmentation models, lightweight SAM variants, classical segmentation baselines, simulation-based robotic datasets, and robotic perception challenges.

---

## 1. Slide-Ready Summary

The current state of the art in segmentation has moved from task-specific supervised models toward **promptable foundation models**. The most important example is **Segment Anything Model (SAM)**, which introduced a general segmentation model trained on the large SA-1B dataset and designed to transfer zero-shot to new images and tasks through prompts such as points and boxes.

After SAM, several related directions became important:

1. **SAM-style foundation segmentation**  
   Models such as SAM and SAM2 aim to provide general-purpose masks from prompts, reducing the need for task-specific retraining.

2. **Video and temporal segmentation**  
   SAM2 extends promptable segmentation from images to videos, which is important for robotics because robot perception is usually continuous and dynamic.

3. **Fast and lightweight variants**  
   FastSAM, MobileSAM, and EfficientSAM try to reduce the computational cost of SAM-style segmentation, making these models more relevant for real-time or edge robotic deployment.

4. **Classical supervised baselines**  
   Mask R-CNN, DeepLabV3+, and YOLOv8-seg remain important because robotic systems often require predictable real-time behavior and can sometimes use a small amount of task-specific labeled data.

5. **Robotic simulation and synthetic data**  
   Isaac Sim and BlenderProc allow controlled generation of robotic scenes with automatic annotations, making it possible to test rare or difficult conditions such as transparent objects, reflective surfaces, occlusions, and small parts.

The research gap is that most segmentation models are not evaluated specifically as **perception modules inside embodied robotic systems**. Robotics requires not only high mask quality, but also robustness, real-time feasibility, and compatibility with perception-action loops.

---

## 2. Foundation Segmentation Models

### 2.1 Segment Anything Model (SAM)

SAM introduced a new way of thinking about segmentation. Instead of training a separate model for every dataset or object category, SAM is designed as a **promptable segmentation model**. A user or another system can provide a point, box, or mask prompt, and the model outputs a segmentation mask for the requested region.

The major contribution of SAM is not only the architecture, but also the scale of training. The authors introduced the **SA-1B dataset**, containing more than one billion masks on eleven million images. This scale allows SAM to generalize to many image distributions and tasks without retraining.

For this project, SAM is important because it represents the first major foundation-model approach to general-purpose segmentation. In robotic perception, this raises a key question: can a robot use a model trained on broad internet-style visual data to segment objects in robot-centered scenes without additional training?

Relevant to this benchmark:

- SAM supports point, box, and mask prompts.
- SAM is designed for zero-shot transfer.
- SAM produces high-quality masks, especially when the object is well indicated by a prompt.
- SAM is computationally heavy, especially with large ViT encoders such as ViT-H.
- SAM does not solve the full robotic perception problem by itself, because the robot still needs to decide what to prompt and when to trust the mask.

**Use in this project:** SAM ViT-H and SAM ViT-B are evaluated as zero-shot promptable segmentation models.

---

### 2.2 Segment Anything Model 2 (SAM2)

SAM2 extends the promptable segmentation idea from static images to both **images and videos**. This is especially relevant for robotics because robots perceive the world over time. A robot does not usually receive isolated images; it receives continuous camera streams while objects, people, and the robot itself may move.

SAM2 introduces a transformer architecture with **streaming memory**, allowing the model to use temporal information during video segmentation. This makes SAM2 more relevant to robotic scenes with dynamic objects, occlusion, and tracking-like requirements.

Relevant to this benchmark:

- SAM2 keeps the promptable segmentation idea.
- SAM2 is designed for both image and video segmentation.
- SAM2 is relevant to dynamic robotic scenes because it can model temporal continuity.
- In this project, SAM2 is used in image-based benchmark mode, but its video-oriented design is important for future robotic extensions.

**Use in this project:** SAM2 Hiera-Large is evaluated as a zero-shot segmentation model.

---

### 2.3 FastSAM

FastSAM is a speed-oriented alternative to SAM. Instead of using the same heavy transformer-based process as SAM, FastSAM reformulates the segment-anything task using an instance-segmentation-style pipeline and prompt-guided mask selection.

The motivation is practical: SAM-style models can be too slow for real-time systems, especially if automatic mask generation is used. FastSAM tries to keep the segment-anything interface while reducing runtime cost.

Relevant to this benchmark:

- FastSAM is useful for testing the speed-quality trade-off.
- It is closer to an instance segmentation pipeline than original SAM.
- It may be more practical for real-time robotic settings, but quality can differ from SAM depending on the scene and prompt mode.

**Use in this project:** FastSAM-X is evaluated as a zero-shot segmentation model.

---

## 3. Lightweight SAM Variants

### 3.1 MobileSAM

MobileSAM addresses the main deployment weakness of SAM: the heavy image encoder. It replaces SAM's large encoder with a lightweight encoder while keeping the general promptable segmentation interface.

This is important for robotics because many robots cannot rely on large desktop GPUs. A perception module may need to run on an onboard computer, embedded GPU, mobile processor, or CPU-constrained system.

Relevant to this benchmark:

- MobileSAM targets smaller and faster deployment.
- It preserves the SAM-style promptable interface.
- It is useful when the robot needs approximate masks faster than full SAM can provide.
- It may sacrifice some segmentation quality compared with heavier SAM models.

**Use in this project:** MobileSAM ViT-T is evaluated as a lightweight SAM-style model.

---

### 3.2 EfficientSAM

EfficientSAM is another attempt to make SAM-style segmentation more efficient. It uses an efficient model design and pretraining strategy to reduce the computational cost while preserving promptable segmentation behavior.

For robotics, EfficientSAM is relevant because edge deployment often requires a compromise between accuracy, speed, memory use, and energy consumption.

Relevant to this benchmark:

- EfficientSAM is designed for efficient promptable segmentation.
- It provides smaller model variants suitable for lightweight comparison.
- It helps evaluate whether foundation segmentation can move from offline experiments toward onboard robotic deployment.

**Use in this project:** EfficientSAM-Ti and EfficientSAM-S are evaluated as lightweight models.

---

## 4. Classical and Supervised Segmentation Baselines

Foundation models are not the only possible solution for robotic segmentation. Classical supervised segmentation models remain important because they can be faster, simpler to deploy, and more predictable once trained on the target domain.

### 4.1 Mask R-CNN

Mask R-CNN is a standard instance segmentation model. It extends Faster R-CNN by adding a mask prediction branch in parallel with object detection. It detects object instances and generates a separate mask for each instance.

Relevant to this benchmark:

- Mask R-CNN is a strong classical instance segmentation baseline.
- It is useful when instance-level object masks are required.
- It requires supervised training or fine-tuning.
- It provides a comparison point against zero-shot foundation models.

**Use in this project:** Mask R-CNN is fine-tuned on a small subset and evaluated as a supervised baseline.

---

### 4.2 DeepLabV3+

DeepLabV3+ is a semantic segmentation architecture that combines multi-scale context with an encoder-decoder structure. Its decoder helps recover sharper object boundaries, which is important when the segmentation quality near object edges matters.

Relevant to this benchmark:

- DeepLabV3+ is a strong semantic segmentation baseline.
- It is useful for category-level segmentation.
- Its boundary refinement makes it relevant to robotic scenes with small or thin objects.
- It requires supervised training or fine-tuning.

**Use in this project:** DeepLabV3+ is fine-tuned on a small subset and evaluated as a supervised semantic segmentation baseline.

---

### 4.3 YOLOv8-seg

YOLOv8-seg is an instance segmentation model from the Ultralytics YOLO family. YOLO models are widely used when real-time object detection or segmentation is needed.

Relevant to this benchmark:

- YOLOv8-seg is useful for real-time robotic perception.
- It can produce instance masks and object detections in one pipeline.
- It requires supervised training or fine-tuning.
- It is an important baseline when speed is as important as accuracy.

**Use in this project:** YOLOv8-seg is fine-tuned on a small subset and evaluated as a real-time supervised baseline.

---

## 5. Simulation and Synthetic Data for Robotic Perception

Robotic perception datasets are difficult to create manually. Real-world annotation is expensive, slow, and sometimes unsafe when robots, moving objects, or fragile materials are involved. Simulation helps because it can generate images and annotations automatically while allowing precise control over objects, materials, lighting, camera positions, and robot poses.

### 5.1 Isaac Sim

Isaac Sim is a robotics simulation platform based on NVIDIA Omniverse. It supports physically based virtual environments, robot simulation, sensor simulation, and synthetic data generation.

For this project, Isaac Sim is important because it allows the benchmark to be centered around a simulated robotic platform, specifically the Unitree G1. This makes the dataset more relevant to robotic perception than generic image datasets.

Relevant to this benchmark:

- Robot-centered simulated scenes.
- Synthetic RGB data and annotation generation.
- Physically based rendering.
- Support for controlled robotic-scene challenges.

**Use in this project:** Isaac Sim is used to generate the official Unitree G1 synthetic robotic-scene dataset.

---

### 5.2 BlenderProc

BlenderProc is a procedural pipeline for photorealistic synthetic data generation. It can generate images and annotations for tasks such as segmentation, depth estimation, optical flow, and object pose estimation.

For this project, BlenderProc is useful as a second simulation pipeline. It provides additional controlled synthetic scenes and supports systematic variation of scene content.

Relevant to this benchmark:

- Procedural synthetic data generation.
- Automatic annotations.
- Controlled object, material, camera, and lighting variation.
- Useful for testing domain gap and robustness.

**Use in this project:** BlenderProc is used to generate the COGAR-SimRobotics synthetic dataset.

---

### 5.3 OCID

OCID, the Object Clutter Indoor Dataset, provides annotated RGB-D cluttered indoor scenes for robot vision. It is relevant because robotic environments often contain cluttered tabletop or indoor scenes where object boundaries are ambiguous and objects may touch or occlude each other.

Relevant to this benchmark:

- Real RGB-D cluttered scenes.
- Object-level annotations.
- Useful for testing whether conclusions from simulation transfer to real robotic-like clutter.

**Use in this project:** OCID is used as a real-world robustness and domain-gap evaluation dataset.

---

## 6. Robotic Perception Challenges Missing from Generic Segmentation Benchmarks

Generic segmentation benchmarks often focus on common object categories and natural images. Robotic perception introduces additional difficulties:

| Robotic challenge | Why it matters |
|---|---|
| Transparent objects | Glass and clear plastic can be difficult for RGB and depth sensors, and their boundaries may be visually weak. |
| Reflective metal | Reflections create misleading textures, highlights, and false boundaries. |
| Partial occlusion | Robots often see objects behind other objects, grippers, robot limbs, or clutter. |
| Small parts | Screws, connectors, thin tools, and cables may occupy very few pixels. |
| Dynamic objects | Moving objects require stable perception over time, not only single-frame accuracy. |
| Robot-body visibility | The robot itself may appear in the camera view and confuse object segmentation. |
| Real-time constraints | A mask is only useful for control if it arrives fast enough for the robot's decision loop. |

These challenges justify why the project does not evaluate segmentation only on standard natural-image datasets. The benchmark is designed around conditions that matter for embodied robotics.

---

## 7. Research Gap Addressed by This Project

The state of the art provides powerful segmentation models, but several questions remain open for robotic perception:

1. **Zero-shot reliability**  
   SAM-style models are trained for broad visual generalization, but robotic scenes can include unusual viewpoints, robot parts, transparent objects, reflective materials, and synthetic-to-real domain shifts.

2. **Prompt dependency**  
   SAM-style models are strongest when a good prompt is available. In robotics, the prompt must come from another process: task context, attention, tracking, detection, grasp planning, or human instruction.

3. **Real-time feasibility**  
   A high-quality mask is not enough if inference is too slow for closed-loop control.

4. **Failure awareness**  
   A cognitive robot must know when its perception is unreliable. Qualitative failure analysis is therefore necessary, not optional.

5. **Model-selection trade-offs**  
   There is no single best segmentation model for all robotic scenarios. A robot may need different models depending on whether the priority is accuracy, speed, edge deployment, or robustness.

This project addresses the gap by systematically benchmarking foundation segmentation models, lightweight variants, and supervised baselines under robotic-scene challenges.

---

## 8. Connection to COGAR

The COGAR relevance of the state of the art is that segmentation is not treated as a final goal. It is treated as a component of a cognitive robotic architecture.

In a cognitive architecture for robotics, perception must support:

- attention,
- object representation,
- spatial reasoning,
- action selection,
- manipulation,
- tracking,
- learning from interaction,
- and failure detection.

SAM-style models are interesting for COGAR because they are **promptable**. A prompt can be interpreted as a form of task-driven attention. For example, a robot may receive a goal such as “pick up the connector,” use a detector or language grounding module to propose a box, and then use a segmentation model to obtain the object mask needed for grasping.

However, the state of the art also shows the limitation: segmentation models are not complete cognitive systems. They do not decide goals, understand tasks, maintain long-term symbolic memory, or guarantee safe action. They provide perceptual structure that must be integrated with higher-level reasoning and control.

A strong COGAR interpretation is therefore:

> Foundation segmentation models can serve as perceptual front-ends for cognitive robotic architectures, but they must be evaluated in terms of robustness, prompt dependency, temporal stability, and real-time deployability before they can support embodied action.

---

## 9. Suggested Slide Content

### Slide 3: State of the Art

Suggested bullets:

- Segmentation research has shifted from task-specific supervised models to promptable foundation models.
- SAM introduced zero-shot promptable segmentation using point, box, and mask prompts.
- SAM2 extends promptable segmentation to images and videos with temporal memory.
- FastSAM, MobileSAM, and EfficientSAM target faster or lighter deployment.
- Mask R-CNN, DeepLabV3+, and YOLOv8-seg remain important supervised baselines for robotics.

Suggested visual:

```text
Supervised segmentation  ->  Promptable foundation segmentation  ->  Robotic deployment trade-offs
Mask R-CNN / DeepLab / YOLO     SAM / SAM2 / FastSAM                 MobileSAM / EfficientSAM / FPS
```

---

### Slide 4: Research Gap

Suggested bullets:

- High benchmark performance on general images does not guarantee robotic reliability.
- Robotic scenes contain transparent, reflective, occluded, small, and moving objects.
- Prompted models require an upstream attention or detection mechanism.
- Robots need masks that are not only accurate, but also fast, stable, and useful for action.
- Therefore, segmentation models must be evaluated as robotic perception modules, not only as image-processing models.

Suggested one-line takeaway:

> The open question is not whether SAM can segment images, but whether SAM-style models can support robust embodied perception.

---

## 10. Speaker Notes

Segmentation is a central perception problem because a robot needs to separate objects from the background before it can reason about them or interact with them. Traditional segmentation systems, such as Mask R-CNN, DeepLabV3+, and YOLOv8-seg, usually require task-specific training. They can be effective and fast, but their performance depends heavily on the training domain.

Foundation segmentation models changed the field by introducing promptable zero-shot segmentation. SAM is the most important example: it can accept points, boxes, or masks and produce segmentation masks without retraining on the target dataset. SAM2 extends this idea to video, which is important because robotic perception is naturally temporal.

However, robotics adds constraints that are not always visible in standard computer vision benchmarks. A robot must handle occlusion, clutter, reflections, transparent objects, small parts, moving objects, and the appearance of its own body. It must also run perception fast enough for action. This is why the state of the art motivates a benchmark that compares not only segmentation quality, but also robustness and speed.

The project therefore evaluates both heavy foundation models and lightweight variants, and compares them with supervised baselines trained on small subsets. This gives a more realistic picture of what kind of segmentation model should be used in different robotic scenarios.

---

## 11. Report-Ready Version

### State of the Art

Recent segmentation research has moved from task-specific supervised architectures toward promptable foundation models. Traditional systems such as Mask R-CNN, DeepLabV3+, and YOLOv8-seg require task-specific training or fine-tuning, but they remain important in robotics because they can provide predictable behavior and high throughput when a representative labeled dataset is available.

The Segment Anything Model introduced a different paradigm: segmentation as a promptable foundation task. SAM is trained on the large SA-1B dataset and accepts prompts such as points, boxes, and masks. This allows the model to transfer zero-shot to new image distributions and segmentation tasks. For robotic perception, SAM is attractive because it can potentially segment previously unseen objects without retraining. However, SAM also introduces prompt dependency: the model must be told what to segment, and a robot must obtain that prompt from another mechanism such as attention, detection, tracking, task context, or human instruction.

SAM2 extends promptable segmentation to images and videos using a transformer architecture with streaming memory. This is especially relevant to robotics because robots perceive continuous streams rather than isolated images. Temporal perception matters for dynamic objects, partial occlusion, object permanence, and tracking. Even when evaluated frame by frame, SAM2 represents an important direction for future robotic perception systems.

Because full SAM-style models can be computationally expensive, several efficient variants have been proposed. FastSAM reformulates segment-anything-style segmentation using a faster instance-segmentation pipeline. MobileSAM reduces the cost of SAM by replacing the heavy image encoder with a lightweight one. EfficientSAM similarly targets efficient promptable segmentation through compact model design and efficient pretraining. These variants are important for robotics because onboard deployment often requires low latency, limited memory use, and acceptable accuracy on embedded hardware.

Simulation-based data generation is also central to modern robotic perception research. Isaac Sim provides physically based robot simulation and synthetic data generation, while BlenderProc provides procedural photorealistic rendering and automatic annotations. These tools make it possible to generate controlled segmentation datasets with known ground truth and targeted challenge categories such as transparent objects, reflective surfaces, occlusions, small parts, and dynamic scenes. OCID provides a complementary real-world cluttered indoor dataset for evaluating robustness beyond simulation.

The remaining gap is that segmentation models are often evaluated as image-processing systems rather than as components of embodied robotic architectures. In robotics, a segmentation mask must support attention, scene representation, tracking, grasping, navigation, or manipulation. Therefore, it must be evaluated not only by mIoU or AP, but also by robustness, boundary quality, inference speed, and failure behavior under robotic-scene conditions. This project addresses that gap by benchmarking SAM-family models, lightweight variants, and supervised baselines on simulated and real robotic scenes.

---

## 12. References

Use these references in the final report and slide speaker notes.

1. Kirillov et al., **Segment Anything**, 2023.  
   https://arxiv.org/abs/2304.02643

2. Ravi et al., **SAM 2: Segment Anything in Images and Videos**, 2024.  
   https://arxiv.org/abs/2408.00714

3. Zhao et al., **Fast Segment Anything**, 2023.  
   https://arxiv.org/abs/2306.12156

4. Zhang et al., **Faster Segment Anything: Towards Lightweight SAM for Mobile Applications**, 2023.  
   https://arxiv.org/abs/2306.14289

5. Xiong et al., **EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything**, 2023.  
   https://arxiv.org/abs/2312.00863

6. He et al., **Mask R-CNN**, 2017.  
   https://arxiv.org/abs/1703.06870

7. Chen et al., **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation**, 2018.  
   https://arxiv.org/abs/1802.02611

8. Ultralytics, **Instance Segmentation Documentation**, accessed 2026.  
   https://docs.ultralytics.com/tasks/segment/

9. NVIDIA, **Isaac Sim**, accessed 2026.  
   https://developer.nvidia.com/isaac/sim

10. Denninger et al., **BlenderProc: Reducing the Reality Gap with Photorealistic Rendering**, 2020.  
    https://arxiv.org/abs/1911.01911

11. OCID, **Object Clutter Indoor Dataset**, accessed 2026.  
    https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-clutter-indoor-dataset/

12. Jiang et al., **Robotic Perception of Transparent Objects: A Review**, 2023.  
    https://ieeexplore.ieee.org/document/10288041

13. University of Genoa, **Cognitive Architectures for Robotics course description**, accessed 2026.  
    https://corsi.unige.it/en/off.f/2023/ins/66538
