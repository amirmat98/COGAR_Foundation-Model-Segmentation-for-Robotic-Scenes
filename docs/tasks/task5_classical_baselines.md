# Task 5: Classical Baseline Models

Task 5 compares zero-shot SAM-family models with supervised classical
segmentation baselines trained on small labeled subsets.

Status: preparation started.

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

- COCO train/validation subset JSON files for Mask R-CNN.
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

## Closure Criteria

Task 5 is complete when YOLOv8-seg, Mask R-CNN, and DeepLabV3+ have trained on
the small subsets and produced prediction files for the benchmark evaluation
stage.
