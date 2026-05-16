# Model Recommendation Guide for Robotic Scene Segmentation

## Dataset and benchmark context

This guide summarizes model recommendations from the final
COGAR-SimRobotics-500 benchmark.

| Property | Value |
|---|---:|
| Images | 500 |
| Object instances | 4,471 |
| Categories | 9 |
| Main metrics | mean IoU, median IoU, boundary F1, per-category IoU, FPS |

The final recommendation uses the completed zero-shot, lightweight, and
supervised baseline results. SAM ViT-H is included only as a CPU subset reference
because full CUDA evaluation is hardware-limited on the available GTX 1050 4 GB.

## Visual evidence

![IoU versus FPS recommendation chart](/outputs/figures/final_report/edge_tradeoff/iou_vs_fps_tradeoff.png)

*Figure: Main recommendation chart showing the accuracy/speed trade-off between promptable, lightweight, speed-first, and supervised options.*

![Failure mode montage](/outputs/figures/final_report/failure_modes/failure_mode_montage.png)

*Figure: Representative failure modes that inform the deployment recommendations and risk notes.*

## Overall full-dataset promptable results

| Model / mode | Objects | Mean IoU | Median IoU | Boundary F1 | IoU >= 0.75 | IoU < 0.10 | Mean FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAM2.1-Tiny box | 4,471 | 0.912746 | 0.955280 | 0.930659 | 0.926191 | 0.000895 | 16.809629 |
| SAM ViT-B box | 4,471 | 0.9057 | 0.9553 | 0.9356 | 0.9137 | 0.0031 | 61.96 |
| EfficientSAM-Ti box | 4,471 | 0.880745 | 0.939880 | 0.910907 | 0.880787 | 0.005592 | 9.474405 |
| MobileSAM box | 4,471 | 0.8656 | 0.9363 | 0.9797 | 0.8430 | 0.0045 | 69.52 |
| FastSAM-S box | 4,471 | 0.698569 | 0.813478 | 0.891956 | 0.592709 | 0.082979 | 471.295280 |

## Prompting recommendation

Box prompts are the most reliable prompt type when object boxes are available.

| Model | Prompt type | Objects | Mean IoU | Median IoU | Boundary F1 | Mean FPS |
|---|---|---:|---:|---:|---:|---:|
| SAM ViT-B | Box | 4,471 | 0.9057 | 0.9553 | 0.9356 | 61.96 |
| SAM ViT-B | Point | 4,471 | 0.7985 | 0.9125 | 0.8131 | not evaluated |
| SAM ViT-B | Auto | 4,471 | 0.8025 | 0.9422 | 0.8381 | not evaluated |
| SAM2.1-Tiny | Box | 4,471 | 0.912746 | 0.955280 | 0.930659 | 16.809629 |
| SAM2.1-Tiny | Point | 4,471 | 0.865783 | 0.934924 | 0.873056 | 16.679109 |
| SAM2.1-Tiny | Auto | 4,471 | 0.640259 | 0.870136 | 0.678148 | 2.300666 |
| FastSAM-S | Box | 4,471 | 0.698569 | 0.813478 | 0.891956 | 471.295280 |
| FastSAM-S | Point | 4,471 | 0.759372 | 0.888325 | 0.788963 | 214.069901 |
| FastSAM-S | Auto / Everything | 4,471 | 0.777331 | 0.891437 | 0.809290 | 206.475290 |

Point prompts are easier to provide but more ambiguous in cluttered scenes.
Automatic mask modes are useful when prompts are unavailable, but they have more
missed-object and proposal-selection failures, especially for small parts and
robot-gripper scenes.

## Recommended model by use case

| Use case | Recommended model | Reason |
|---|---|---|
| Highest promptable mean IoU | SAM2.1-Tiny box | Best completed full-dataset box-prompt mean IoU |
| Best speed/accuracy balance for prompted masks | SAM ViT-B box | High accuracy with much higher measured FPS than SAM2.1-Tiny |
| Edge deployment / lightweight SAM-style model | MobileSAM box | Best lightweight SAM-style edge trade-off |
| Lightweight accuracy when speed is less important | EfficientSAM-Ti box | Strong IoU, but slower measured FPS than MobileSAM |
| Speed-first zero-shot approximate masks | FastSAM-S box | Highest measured SAM-style throughput |
| Automatic supervised deployment | YOLOv8n-seg | Real-time feed-forward instance segmentation after fine-tuning |
| Classical supervised comparison | Mask R-CNN ResNet-50 FPN | Standard instance-segmentation baseline |
| Large SAM reference | SAM ViT-H | Valid CPU subset only; full CUDA hardware-limited |

## Supervised baseline recommendation

| Model | Final role | Main result |
|---|---|---|
| YOLOv8n-seg | Main supervised deployment baseline | mask mAP50 0.806, mask mAP50-95 0.601, about 37.3 FPS total |
| Mask R-CNN ResNet-50 FPN | Classical supervised instance baseline | mean IoU 0.7462, median IoU 0.8309, 5.5855 image FPS |
| DeepLabV3+ | Excluded | Semantic segmentation model, not directly comparable to instance-level masks |

YOLOv8n-seg is recommended when prompts are unavailable and supervised training
on the target domain is allowed. Mask R-CNN is useful as a classical comparison
point but is slower on the available hardware. DeepLabV3+ is excluded because it
predicts semantic class regions rather than separate object-instance masks.

## Category-specific recommendations

| Category | Main difficulty | Recommended model |
|---|---|---|
| box | Large regular object, usually easy | SAM2.1-Tiny, SAM ViT-B, or MobileSAM |
| cable | Thin elongated geometry | SAM2.1-Tiny or SAM ViT-B |
| connector | Small part | SAM2.1-Tiny, SAM ViT-B, or MobileSAM |
| glass_object | Transparent boundaries | SAM2.1-Tiny or SAM ViT-B |
| metal_part | Reflective material | SAM2.1-Tiny or SAM ViT-B |
| plastic_object | Usually easier object | SAM2.1-Tiny, SAM ViT-B, or MobileSAM |
| robot_gripper | Articulated shape, holes, contact with objects | SAM2.1-Tiny or SAM ViT-B |
| screw | Very small object | SAM2.1-Tiny or SAM ViT-B |
| tool | Thin/complex shape | SAM2.1-Tiny or SAM ViT-B |

## Challenge-specific recommendations

| Challenge | Recommended model | Notes |
|---|---|---|
| reflective_metal | SAM2.1-Tiny box or SAM ViT-B box | Strongest promptable options |
| transparent_glass | SAM2.1-Tiny box or SAM ViT-B box | Still a high-risk challenge family |
| partial_occlusion | SAM2.1-Tiny box or SAM ViT-B box | Prompted masks are preferred |
| small_parts | SAM2.1-Tiny box or SAM ViT-B box | Avoid relying only on automatic masks |
| dynamic_scene | SAM ViT-B box or MobileSAM box | Balance accuracy and runtime for manipulation scenes |

## Failure-mode summary

The qualitative failure analysis shows that the hardest cases are:

- robot grippers and articulated robot parts
- cables and thin structures
- screws, connectors, and other small parts
- reflective metal parts
- transparent glass objects
- partial occlusions
- dynamic clutter around the robot
- prompt ambiguity in crowded scenes
- automatic-mask proposal misses

Robot grippers dominate several worst-case examples for SAM ViT-B and MobileSAM.
FastSAM-S has more severe failures on metal parts, tools, screws, and
small-part scenes.

## Hardware recommendation

| Model | Practical status on GTX 1050 4 GB |
|---|---|
| SAM ViT-B | Practical full benchmark model |
| SAM2.1-Tiny | Accurate and feasible, but slower than SAM ViT-B |
| SAM ViT-H | Valid checkpoint, CPU subset only |
| MobileSAM | Practical lightweight model |
| EfficientSAM-Ti | Accurate lightweight model, slower measured FPS |
| FastSAM-S | Practical high-speed model |
| YOLOv8n-seg | Practical real-time supervised model |
| Mask R-CNN | Useful baseline, not real-time |

## Final recommendation

Use SAM2.1-Tiny box when maximum promptable accuracy is the priority and lower
speed is acceptable. Use SAM ViT-B box when a stronger speed/accuracy balance is
needed. Use MobileSAM for the best lightweight SAM-style edge trade-off. Use
FastSAM-S when speed matters more than mask quality. Use YOLOv8n-seg when
supervised automatic segmentation is allowed and prompts are unavailable.

Treat transparent objects, reflective objects, cables, small parts, grippers,
partial occlusion, and dynamic manipulation clutter as high-risk cases in any
robotic deployment.
