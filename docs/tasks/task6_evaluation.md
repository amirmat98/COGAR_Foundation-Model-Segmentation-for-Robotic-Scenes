# Task 6: Evaluation Metrics

Task 6 evaluates the zero-shot SAM-family predictions and the supervised
baseline outputs with a common metric layer.

## Common Held-Out Test Protocol

Final comparisons use the test IDs created before supervised training. The
100-image training and 50-image validation subsets are never used for final
quality tables. Validation is used only for baseline checkpoint selection.

| Dataset | Train | Validation | Final test |
| --- | ---: | ---: | ---: |
| Isaac official Unitree G1 | 100 | 50 | 850 |
| BlenderProc COGAR-SimRobotics-1000 | 100 | 50 | 850 |
| OCID | 100 | 50 | 2240 |

Both zero-shot and supervised evaluators read the same
`outputs/task5_baselines/splits/<dataset>/test_image_ids.txt` files. Every
summary row records `split=test`, `evaluation_images`, and the SHA256 digest of
the test-ID file. Final asset generation rejects non-test rows, mismatched image
counts, or different test-ID digests.

## Figures

![Zero-shot mIoU heatmap](../../outputs/final_benchmark_assets/plots/zero_shot_miou_heatmap.png)

![Classical baseline mIoU by dataset](../../outputs/final_benchmark_assets/plots/baseline_miou_bars.png)

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

## Full Corrected Run

The Task 4/Task 9 prediction JSONL files and the three saved baseline
checkpoints must be present on the GPU machine. The split itself is unchanged,
so baseline retraining is not required.

First regenerate the ignored COCO derivatives:

```bash
python scripts/datasets/convert_ocid_to_coco.py \
  --output-file outputs/datasets/ocid/instances_all.json \
  --metadata-file outputs/datasets/ocid/frame_index.csv \
  --splits-dir outputs/datasets/ocid/splits

python scripts/baselines/prepare_task5_splits.py --formats coco
```

Then run the common-test evaluation:

```bash
python scripts/evaluation/evaluate_task6_zero_shot.py \
  --split test \
  --rerun-complete

python scripts/evaluation/evaluate_task6_baselines.py \
  --split test \
  --device 0 \
  --rerun-complete

python scripts/evaluation/evaluate_task6_zero_shot.py \
  --config configs/task9_evaluation.yaml \
  --split test \
  --rerun-complete

python scripts/analysis/summarize_task9_lightweight_sam.py
python scripts/analysis/create_final_benchmark_assets.py
```

The last two commands deliberately fail until all common-test summaries exist
and contain identical split hashes and image counts.

## Outputs

Compact evaluation summaries are written under:

```text
outputs/task6_evaluation/zero_shot/test
outputs/task6_evaluation/baselines/test
```

The older summaries directly under `zero_shot/` and `baselines/` are legacy
full-dataset/validation results. They must not be used in final comparative
tables.

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
  --split test \
  --max-records 5 \
  --rerun-complete
```

Use a small baseline check:

```text
python scripts/evaluation/evaluate_task6_baselines.py \
  --baselines mask_rcnn deeplabv3plus \
  --datasets blenderproc_cogar_sim \
  --split test \
  --rerun-complete
```

YOLOv8-seg, Mask R-CNN, and DeepLabV3+ load the checkpoint selected on the
validation split and perform fresh inference on the held-out test split.
