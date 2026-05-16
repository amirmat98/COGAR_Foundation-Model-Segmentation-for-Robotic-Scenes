# Assignment Task 7: Inference Speed and Real-Time Feasibility

## Task requirement

Measure inference speed, reported as FPS, on GPU and CPU to assess real-time feasibility for robotic applications.

## Status

Task 7 is completed with a hardware-aware interpretation.

GPU inference speed was measured for the completed full-dataset or full-test evaluations where feasible. CPU inference speed was measured using SAM ViT-H subset runs, because full CPU evaluation of large foundation models is too slow for the 4,471-object benchmark and not suitable for real-time robotic deployment.

## Hardware

| Component | Value |
|---|---|
| GPU | NVIDIA GTX 1050 |
| GPU memory | 4 GB VRAM |
| CPU setting | CPU-only fallback / subset proof runs |
| Dataset | COGAR-SimRobotics-500 |
| Full zero-shot object count | 4,471 objects |
| Supervised test split | 75 images / 679 objects |

## GPU inference speed summary

| Model | Mode | Device | Evaluation size | FPS | FPS type | Real-time feasibility |
|---|---|---|---:|---:|---|---|
| SAM ViT-B | Box | CUDA | 4,471 objects | 61.96 | object-row-level | Feasible when box prompts are available |
| MobileSAM | Box | CUDA | 4,471 objects | 69.52 | object-row-level | Feasible lightweight SAM-style option |
| FastSAM-S | Box | CUDA | 4,471 objects | 471.30 | object-row-level | Strongest speed-first zero-shot option |
| FastSAM-S | Point | CUDA | 4,471 objects | 214.07 | object-row-level | Real-time feasible |
| FastSAM-S | Auto / Everything | CUDA | 4,471 objects | 206.48 | object-row-level | Real-time feasible |
| EfficientSAM-Ti | Box | CUDA | 4,471 objects | 9.47 | object-row-level | Borderline / slower lightweight option |
| SAM2.1-Tiny | Box | CUDA | 4,471 objects | 16.81 | object-row-level | Potentially usable, but below high-FPS real-time |
| SAM2.1-Tiny | Point | CUDA | 4,471 objects | 16.68 | object-row-level | Potentially usable, but below high-FPS real-time |
| SAM2.1-Tiny | Auto | CUDA | 4,471 objects | 2.30 | object-row-level | Not real-time on GTX 1050 |
| YOLOv8n-seg | Automatic supervised | CUDA | 75 test images | 37.3 | total image FPS | Real-time feasible |
| YOLOv8n-seg | Automatic supervised | CUDA | 75 test images | 90.1 | inference-only image FPS | Real-time feasible |
| Mask R-CNN R50-FPN | Automatic supervised | CUDA | 75 test images | 5.59 | image FPS | Not real-time on this hardware |

## CPU inference speed summary

| Model | Mode | Device | Evaluation size | FPS | Interpretation |
|---|---|---|---:|---:|---|
| SAM ViT-H | Box | CPU | 25 objects | 0.1820 | Too slow for real-time robotic use |
| SAM ViT-H | Point | CPU | 25 objects | 0.1762 | Too slow for real-time robotic use |
| SAM ViT-H | Auto | CPU | 42 objects / 5 images | 0.2118 | Too slow for real-time robotic use |

SAM ViT-H could not be evaluated on CUDA because the GTX 1050 4 GB GPU ran out of memory during image encoding. CPU evaluation confirms implementation correctness, but CPU FPS is far below real-time requirements.

## Important timing note

For promptable models evaluated object-by-object, the reported FPS values are object-row-level timing values from the benchmark scripts. Several objects can come from the same image, and some implementations may reuse image-level computation. Therefore, these FPS values should be interpreted as relative throughput indicators, not strict end-to-end robot camera-frame FPS.

For YOLOv8n-seg and Mask R-CNN, FPS is image-level on the supervised test split, which is closer to deployment-style runtime because these models process a full image and directly output instance masks.

## Real-time threshold used

For this project, approximately 30 FPS is treated as the practical real-time target for robotic camera-frame processing. Models clearly above this threshold are considered real-time feasible on the available hardware. Models far below this threshold are treated as unsuitable for real-time deployment on the tested laptop GPU/CPU.

## Real-time feasibility ranking

### Most feasible for real-time robotic use

1. YOLOv8n-seg
2. FastSAM-S
3. MobileSAM
4. SAM ViT-B box prompting

YOLOv8n-seg is the most practical automatic deployment baseline because it does not require prompts and reaches real-time image-level throughput on the GTX 1050.

FastSAM-S is the fastest promptable/automatic-mask family model, but with lower segmentation accuracy than SAM ViT-B and SAM2.1-Tiny.

MobileSAM and SAM ViT-B are feasible when object proposals or bounding-box prompts are available.

### Borderline or not real-time

- SAM2.1-Tiny box/point is accurate but slower on GTX 1050.
- SAM2.1-Tiny auto is not real-time on GTX 1050.
- Mask R-CNN is not real-time on this hardware.
- SAM ViT-H CPU is not real-time and CUDA is infeasible due to memory limits.

## Conclusion

Task 7 is completed.

The benchmark includes GPU FPS measurements for the completed full evaluations and CPU FPS measurements for the hardware-limited SAM ViT-H subset runs. The results show that real-time robotic feasibility depends strongly on model type and prompting mode. YOLOv8n-seg and FastSAM-S are the strongest speed-first options, while SAM2.1-Tiny and SAM ViT-B provide stronger accuracy trade-offs. SAM ViT-H and Mask R-CNN are not practical for real-time use on the available GTX 1050 4 GB hardware.
