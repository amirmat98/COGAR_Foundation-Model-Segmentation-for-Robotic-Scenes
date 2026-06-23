# Task 5: Classical Baseline Models

Task 5 compares zero-shot SAM-family models with supervised classical
segmentation baselines trained on small labeled subsets.

Status: complete.

## Figure

![Classical baseline mIoU by dataset](../../outputs/final_benchmark_assets/plots/baseline_miou_bars.png)

## Baselines

The baseline set is:

- YOLOv8-seg for instance segmentation.
- Mask R-CNN for instance segmentation.
- DeepLabV3+ for semantic segmentation.

YOLOv8-seg and Mask R-CNN produce instance masks. DeepLabV3+ produces semantic
class masks, so its evaluation is reported for semantic metrics rather than
instance AP.

## Small Subset Policy

Each dataset uses a deterministic small supervised split:

- 100 training images.
- 50 validation images.
- Remaining images kept for benchmark testing/evaluation.

The split seed is `5884715`.

## Prepared Formats

The preparation script writes:

- COCO train/validation files for training plus a held-out test COCO file used
  only by Task 6 evaluation.
- YOLOv8-seg image/label folders and dataset YAML files.
- DeepLabV3+ image folders, semantic mask folders, and dataset YAML files.

Compact configs and summaries are written under:

```text
outputs/task5_baselines
```

Generated training folders are written under:

```text
data/task5_baselines
```

The generated training folders are local artifacts and are not committed to
Git.

## Preparation Command

Use:

```text
python scripts/baselines/prepare_task5_splits.py
```

The command only prepares small supervised training data. It does not train any
model.

## YOLOv8-Seg Training

YOLOv8-seg is the first supervised baseline because it is the simplest
instance-segmentation baseline to fine-tune on the prepared subsets.

The first configured model is:

```text
yolov8n-seg.pt
```

Training uses the 100-image train split and 50-image validation split generated
by Task 5A. Training outputs are written under:

```text
results/task5_baselines/yolo8_seg
```

Compact training summaries are written under:

```text
outputs/task5_baselines/yolo8_seg_training
```

Use a smoke run first:

```text
python scripts/baselines/train_yolov8_seg.py --datasets blenderproc_cogar_sim --smoke
```

Then run all enabled datasets:

```text
python scripts/baselines/train_yolov8_seg.py
```

After training, collect compact validation metrics from the YOLOv8 run folders:

```text
python scripts/baselines/collect_yolov8_seg_metrics.py
```

This writes `metrics_summary.json` and `metrics_summary.csv` under
`outputs/task5_baselines/yolo8_seg_training`.

## Mask R-CNN Training

Mask R-CNN is trained with TorchVision's `maskrcnn_resnet50_fpn` implementation
initialized from COCO weights. It uses the COCO train/validation subset files
prepared in Task 5A.

This covers the requested Mask R-CNN baseline family while keeping the training
pipeline in the same PyTorch/TorchVision environment used by the rest of the
project. Detectron2 was not required for the final baseline because it would
provide another implementation of the same Mask R-CNN model family rather than
a distinct baseline category.

Training outputs are written under:

```text
results/task5_baselines/mask_rcnn
```

Compact training summaries and validation AP metrics are written under:

```text
outputs/task5_baselines/mask_rcnn_training
```

Use a smoke run first:

```text
python scripts/baselines/train_mask_rcnn.py --datasets blenderproc_cogar_sim --smoke
```

Then run all enabled datasets:

```text
python scripts/baselines/train_mask_rcnn.py
```

## DeepLabV3+ Training

DeepLabV3+ is trained with `segmentation-models-pytorch` using a ResNet-34
encoder initialized from ImageNet weights. It uses the semantic image/mask
folders prepared in Task 5A.

Training outputs are written under:

```text
results/task5_baselines/deeplabv3plus
```

Compact training summaries and validation semantic metrics are written under:

```text
outputs/task5_baselines/deeplabv3plus_training
```

Use a smoke run first:

```text
python scripts/baselines/train_deeplabv3plus.py --datasets blenderproc_cogar_sim --smoke
```

Then run all enabled datasets:

```text
python scripts/baselines/train_deeplabv3plus.py
```

## Closure Criteria

Task 5 is complete when YOLOv8-seg, Mask R-CNN, and DeepLabV3+ have trained on
the small subsets and produced prediction files for the benchmark evaluation
stage.

The completed baseline runs produced compact summaries under:

```text
outputs/task5_baselines/yolo8_seg_training
outputs/task5_baselines/mask_rcnn_training
outputs/task5_baselines/deeplabv3plus_training
```

Task 6 selects checkpoints using validation metrics, then performs fresh
inference on the held-out test IDs shared with every zero-shot model.
