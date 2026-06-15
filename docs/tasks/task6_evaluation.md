# Task 6: Evaluation Metrics

Task 6 evaluates the zero-shot SAM-family predictions and the supervised
baseline outputs with a common metric layer.

## Metrics

The reported metrics are:

- `mIoU`: mean mask IoU over matched ground-truth instances.
- `boundary_f1`: boundary F1 with a 2-pixel tolerance.
- `mask_AP`, `mask_AP50`, `mask_AP75`: COCO mask AP for instance predictions.
- `per_category_iou`: mean IoU grouped by ground-truth category.

For automatic zero-shot masks, AP is class-agnostic because the models produce
category-free mask proposals. All categories are mapped to a single `object`
class for this AP calculation, while per-category IoU is still grouped by the
ground-truth category.

For DeepLabV3+, mask AP is not applicable because the model produces semantic
class maps rather than instance masks. It is evaluated with semantic mIoU,
foreground mIoU, boundary F1, and per-category IoU.

## Scripts

Zero-shot SAM-family evaluation:

```text
python scripts/evaluation/evaluate_task6_zero_shot.py
```

Supervised baseline evaluation:

```text
python scripts/evaluation/evaluate_task6_baselines.py
```

Both scripts read [configs/task6_evaluation.yaml](../../configs/task6_evaluation.yaml).

## Outputs

Compact evaluation summaries are written under:

```text
outputs/task6_evaluation
```

Large generated prediction files, such as YOLOv8 validation predictions, are
written under:

```text
results/task6_evaluation
```

The `results` directory is ignored by Git and should be treated as a local or
release artifact.

## Smoke Checks

Use a small zero-shot check before the full run:

```text
python scripts/evaluation/evaluate_task6_zero_shot.py \
  --datasets blenderproc_cogar_sim \
  --models sam_vit_b \
  --prompt-modes point automatic \
  --max-records 5 \
  --rerun-complete
```

Use a small baseline check:

```text
python scripts/evaluation/evaluate_task6_baselines.py \
  --baselines mask_rcnn deeplabv3plus \
  --datasets blenderproc_cogar_sim \
  --rerun-complete
```

YOLOv8-seg baseline evaluation loads the saved YOLO weights and performs
validation-set inference if the Task 6 prediction JSON file does not already
exist.
