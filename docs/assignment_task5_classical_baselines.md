# Assignment Task 5: Classical and Supervised Baseline Models

## Task requirement

Run classical baseline models such as Mask R-CNN, DeepLabV3+, and YOLOv8-seg fine-tuned on a small subset for comparison.

## Status

Task 5 is completed with two instance-level supervised segmentation baselines:

1. YOLOv8n-seg fine-tuned on a small COGAR-Sim training subset.
2. Mask R-CNN ResNet-50 FPN fine-tuned/evaluated on the COGAR-Sim supervised split.

DeepLabV3+ was reviewed but excluded from the final instance-level benchmark because it is a semantic segmentation model, while this project evaluates object instances using instance masks.

## Baseline summary

| Model | Type | Training / evaluation status | Main metrics reported | Final role |
|---|---|---|---|---|
| YOLOv8n-seg | Supervised instance segmentation | Completed | Mask precision, recall, mAP50, mAP50-95, FPS, per-class mask mAP | Main real-time supervised baseline |
| Mask R-CNN ResNet-50 FPN | Supervised instance segmentation | Completed | Mean IoU, median IoU, boundary F1, per-class IoU, FPS | Classical supervised instance baseline |
| DeepLabV3+ | Semantic segmentation | Excluded | Not evaluated as instance segmentation | Documented limitation / not directly comparable |

## Visual evidence

![Supervised baseline summary](/outputs/figures/final_report/metrics/supervised_baselines_summary.png)

*Figure: YOLOv8n-seg AP metrics and Mask R-CNN IoU/BF1 metrics, shown separately because AP and IoU are different metric families.*

![FPS comparison with supervised baselines context](/outputs/figures/final_report/speed/fps_comparison.png)

*Figure: Speed context for segmentation models and baselines. YOLOv8n-seg image-level speed is reported in the tables below.*

## YOLOv8n-seg fine-tuned baseline

YOLOv8n-seg was fine-tuned as the practical supervised deployment baseline.

### Dataset split

| Split | Images |
|---|---:|
| Train | 120 |
| Validation | 75 |
| Test | 75 |

Test instances: 679

### Training setup

| Item | Value |
|---|---|
| Model | `yolov8n-seg.pt` |
| Epochs | 30 |
| Image size | 640 |
| Batch size | 4 |
| GPU | NVIDIA GTX 1050 4 GB |

### Test results

| Metric | Value |
|---|---:|
| Mask precision | 0.761 |
| Mask recall | 0.783 |
| Mask mAP50 | 0.806 |
| Mask mAP50-95 | 0.601 |
| Total speed | 26.8 ms/image |
| Total FPS | 37.3 |
| Inference-only FPS | 90.1 |

### Interpretation

YOLOv8n-seg is the strongest deployment-style supervised baseline because it performs automatic instance segmentation without prompts and reaches real-time image-level speed on the available GPU.

It depends on supervised training data, so it is less zero-shot than SAM-style methods, but it is practically useful when the object categories and visual domain are known.

Supporting file:

- `docs/final_yolov8seg_baseline_results.md`

## Mask R-CNN ResNet-50 FPN baseline

Mask R-CNN was added as a classical supervised instance-segmentation baseline.

### Training / evaluation setup

| Item | Value |
|---|---|
| Model | Mask R-CNN ResNet-50 FPN |
| Implementation | TorchVision |
| Pretraining | COCO weights |
| Train split | Full supervised training split |
| Test split | 75 images / 679 objects |
| Epochs | 5 |
| Batch size | 1 |
| Image size | min-size 320, max-size 512 |
| Device | NVIDIA GTX 1050 4 GB |
| AMP | Enabled |
| Backbone | Frozen |
| Score threshold | 0.05 |

### Test results

| Metric | Value |
|---|---:|
| Test images | 75 |
| Test objects | 679 |
| Mean IoU | 0.7462 |
| Median IoU | 0.8309 |
| Mean boundary F1 | 0.7218 |
| IoU >= 0.90 | 0.2872 |
| IoU >= 0.75 | 0.6745 |
| IoU >= 0.50 | 0.8748 |
| IoU < 0.10 | 0.0604 |
| Total inference time | 13.4276 s |
| Image FPS | 5.5855 |
| Object FPS | 50.5675 |

### Per-class Mask R-CNN IoU

| Class | Objects | Mean IoU | Median IoU | IoU >= 0.50 | IoU < 0.10 |
|---|---:|---:|---:|---:|---:|
| box | 54 | 0.8124 | 0.9025 | 0.9074 | 0.0370 |
| cable | 42 | 0.5602 | 0.6104 | 0.6905 | 0.0952 |
| connector | 86 | 0.7575 | 0.8611 | 0.8721 | 0.0698 |
| glass_object | 54 | 0.6852 | 0.7897 | 0.8148 | 0.1296 |
| metal_part | 78 | 0.8288 | 0.8907 | 0.9359 | 0.0256 |
| plastic_object | 97 | 0.8499 | 0.9262 | 0.9381 | 0.0515 |
| robot_gripper | 155 | 0.7487 | 0.8066 | 0.9097 | 0.0258 |
| screw | 66 | 0.6396 | 0.7590 | 0.8333 | 0.1364 |
| tool | 47 | 0.6755 | 0.7216 | 0.7872 | 0.0426 |

### Interpretation

Mask R-CNN provides a useful classical supervised comparison point. It achieves reasonable instance-mask IoU, but it is slower than YOLOv8n-seg and not real-time on the GTX 1050.

Its weakest categories are cable, screw, tool, and glass_object, matching the broader project trend that thin, small, transparent, reflective, and irregular objects are difficult.

Supporting file:

- `docs/final_maskrcnn_baseline_results.md`

## DeepLabV3+ exclusion note

DeepLabV3+ was not included in the final quantitative baseline table because it is designed for semantic segmentation.

The benchmark in this project is instance-level: every object instance has its own mask and is evaluated with instance IoU, boundary F1, per-category IoU, and mask AP where applicable.

A semantic segmentation model predicts class regions, not separate masks for each object instance. Therefore, DeepLabV3+ is not directly comparable to SAM, SAM2, FastSAM, YOLOv8-seg, and Mask R-CNN under the current instance-level protocol.

This is documented as a scope limitation rather than an implementation failure.

## Comparison with zero-shot models

The supervised baselines serve a different role from the promptable zero-shot models:

| Family | Requires training on COGAR-Sim? | Requires prompt? | Produces automatic instance masks? | Main use |
|---|---:|---:|---:|---|
| SAM / SAM2 | No | Yes for box/point; no for auto mode | Partly | Zero-shot promptable segmentation |
| FastSAM | No | Optional depending on mode | Yes in everything/auto mode | Fast zero-shot segmentation |
| YOLOv8n-seg | Yes | No | Yes | Real-time supervised deployment baseline |
| Mask R-CNN | Yes | No | Yes | Classical supervised comparison |
| DeepLabV3+ | Yes | No | No, semantic regions only | Excluded from instance benchmark |

## Conclusion

Task 5 is completed.

YOLOv8n-seg and Mask R-CNN provide supervised instance-segmentation baselines for comparison with the zero-shot foundation models. DeepLabV3+ is documented as excluded because the project benchmark is instance-level, while DeepLabV3+ is semantic segmentation and therefore not directly comparable under the current evaluation protocol.
