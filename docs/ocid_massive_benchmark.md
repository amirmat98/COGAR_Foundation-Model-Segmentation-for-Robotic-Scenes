# OCID Massive Benchmark

This workflow runs a large real-world OCID generalization benchmark using the
local dataset at:

```text
/mnt/Info/COGAR_DATASETs/OCID-dataset
```

It is separate from the final simulated COGAR-SimRobotics-500 assignment
benchmark. Use it as an additional robustness/generalization experiment.

For EC2 GPU runs, use the AWS wrapper workflow in
`docs/aws_ocid_benchmark.md`.

For Azure Windows GPU VMs, use the PowerShell workflow in
`docs/azure_windows_ocid_benchmark.md`.

## 1. Build the full OCID object index

```bash
PYTHONPATH=src python3 scripts/dataset/create_object_index.py \
  --dataset ocid_full \
  --config configs/paths.yaml \
  --ocid-root /mnt/Info/COGAR_DATASETs/OCID-dataset \
  --progress-every 250
```

This scans all OCID sequences, creates image/object indexes, filters noisy
objects, and exports one binary mask per object.

Outputs:

```text
outputs/ocid_full/indexes/ocid_full_images.csv
outputs/ocid_full/indexes/ocid_full_objects.csv
outputs/ocid_full/indexes/ocid_full_objects_filtered.csv
outputs/ocid_full/indexes/ocid_full_objects_filtered_with_masks.csv
outputs/ocid_full/gt_binary_masks/
```

Optional filter tuning:

```bash
PYTHONPATH=src python3 scripts/dataset/create_object_index.py \
  --dataset ocid_full \
  --config configs/paths.yaml \
  --ocid-root /mnt/Info/COGAR_DATASETs/OCID-dataset \
  --ocid-min-area 500 \
  --ocid-max-area-ratio 0.08 \
  --ocid-max-bbox-area-ratio 0.15
```

Debug/strict mode:

```bash
PYTHONPATH=src python3 scripts/dataset/create_object_index.py \
  --dataset ocid_full \
  --config configs/paths.yaml \
  --ocid-root /mnt/Info/COGAR_DATASETs/OCID-dataset \
  --progress-every 50 \
  --debug \
  --strict
```

Use `--debug` when the command seems stuck and `--strict` when you want missing
or unreadable labels to stop the run immediately instead of being skipped.

## 2. Smoke-test model runs

Run these first before launching full OCID jobs.

SAM ViT-B box prompts:

```bash
PYTHONPATH=src python3 scripts/eval/run_sam_box_prompt.py \
  --config configs/paths.yaml \
  --index outputs/ocid_full/indexes/ocid_full_objects_filtered_with_masks.csv \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --device auto \
  --limit 25 \
  --output-dir outputs/ocid_full/smoke/sam_vit_b_box \
  --results-csv outputs/ocid_full/results/smoke_sam_vit_b_box.csv \
  --no-visualizations \
  --progress-every 1
```

SAM ViT-B point prompts:

```bash
PYTHONPATH=src python3 scripts/eval/run_sam_point_prompt.py \
  --config configs/paths.yaml \
  --index outputs/ocid_full/indexes/ocid_full_objects_filtered_with_masks.csv \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --device auto \
  --limit 25 \
  --output-dir outputs/ocid_full/smoke/sam_vit_b_point \
  --results-csv outputs/ocid_full/results/smoke_sam_vit_b_point.csv \
  --no-visualizations \
  --progress-every 1
```

SAM ViT-B automatic masks:

```bash
PYTHONPATH=src python3 scripts/eval/run_sam_auto_masks.py \
  --config configs/paths.yaml \
  --index outputs/ocid_full/indexes/ocid_full_objects_filtered_with_masks.csv \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --device auto \
  --limit 10 \
  --output-dir outputs/ocid_full/smoke/sam_vit_b_auto \
  --results-csv outputs/ocid_full/results/smoke_sam_vit_b_auto.csv \
  --progress-every 1
```

## 3. Full SAM ViT-B OCID runs

Use these after smoke tests pass.

```bash
PYTHONPATH=src python3 scripts/eval/run_sam_box_prompt.py \
  --config configs/paths.yaml \
  --index outputs/ocid_full/indexes/ocid_full_objects_filtered_with_masks.csv \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --device auto \
  --output-dir outputs/ocid_full/sam_vit_b_box \
  --results-csv outputs/ocid_full/results/sam_vit_b_box.csv \
  --no-visualizations \
  --no-save-masks \
  --progress-every 500
```

```bash
PYTHONPATH=src python3 scripts/eval/run_sam_point_prompt.py \
  --config configs/paths.yaml \
  --index outputs/ocid_full/indexes/ocid_full_objects_filtered_with_masks.csv \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --device auto \
  --output-dir outputs/ocid_full/sam_vit_b_point \
  --results-csv outputs/ocid_full/results/sam_vit_b_point.csv \
  --no-visualizations \
  --no-save-masks \
  --progress-every 500
```

```bash
PYTHONPATH=src python3 scripts/eval/run_sam_auto_masks.py \
  --config configs/paths.yaml \
  --index outputs/ocid_full/indexes/ocid_full_objects_filtered_with_masks.csv \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --device auto \
  --output-dir outputs/ocid_full/sam_vit_b_auto_fast16 \
  --results-csv outputs/ocid_full/results/sam_vit_b_auto_fast16.csv \
  --points-per-side 16 \
  --pred-iou-thresh 0.90 \
  --stability-score-thresh 0.92 \
  --no-save-masks \
  --progress-every 500
```

The automatic-mask command above is the recommended AWS/default command. To run
SAM automatic masks with Segment Anything's built-in dense defaults, remove the
`--points-per-side`, `--pred-iou-thresh`, and `--stability-score-thresh`
arguments and use a separate results CSV such as
`outputs/ocid_full/results/sam_vit_b_auto.csv`.

## 4. Optional additional model runs

FastSAM-S box:

```bash
PYTHONPATH=src python3 scripts/eval/run_fastsam_box_prompt.py \
  --index outputs/ocid_full/indexes/ocid_full_objects_filtered_with_masks.csv \
  --checkpoint checkpoints/FastSAM-s.pt \
  --device auto \
  --output-dir outputs/ocid_full/fastsam_s_box \
  --results-csv outputs/ocid_full/results/fastsam_s_box.csv
```

MobileSAM box:

```bash
PYTHONPATH=src python3 scripts/eval/run_mobilesam_box_prompt.py \
  --index outputs/ocid_full/indexes/ocid_full_objects_filtered_with_masks.csv \
  --checkpoint checkpoints/mobile_sam.pt \
  --device auto \
  --output-dir outputs/ocid_full/mobilesam_box \
  --results-csv outputs/ocid_full/results/mobilesam_box.csv
```

FastSAM-S point and automatic masks:

```bash
PYTHONPATH=src python3 scripts/foundation_models/evaluate_fastsam_prompt_modes.py \
  --index-csv outputs/ocid_full/indexes/ocid_full_objects_filtered_with_masks.csv \
  --model checkpoints/FastSAM-s.pt \
  --output-dir outputs/ocid_full/fastsam_s_point \
  --prompt-type point \
  --device cuda
```

```bash
PYTHONPATH=src python3 scripts/foundation_models/evaluate_fastsam_prompt_modes.py \
  --index-csv outputs/ocid_full/indexes/ocid_full_objects_filtered_with_masks.csv \
  --model checkpoints/FastSAM-s.pt \
  --output-dir outputs/ocid_full/fastsam_s_auto \
  --prompt-type auto \
  --device cuda
```

SAM2.1 Tiny box, point, and automatic masks:

```bash
PYTHONPATH=src python3 scripts/foundation_models/evaluate_sam2_box_point.py \
  --index-csv outputs/ocid_full/indexes/ocid_full_objects_filtered_with_masks.csv \
  --sam2-repo /path/to/sam2 \
  --checkpoint checkpoints/sam2.1_hiera_tiny.pt \
  --config configs/sam2.1/sam2.1_hiera_t.yaml \
  --output-dir outputs/ocid_full/sam2_tiny_box \
  --prompt-type box \
  --device cuda
```

```bash
PYTHONPATH=src python3 scripts/foundation_models/evaluate_sam2_box_point.py \
  --index-csv outputs/ocid_full/indexes/ocid_full_objects_filtered_with_masks.csv \
  --sam2-repo /path/to/sam2 \
  --checkpoint checkpoints/sam2.1_hiera_tiny.pt \
  --config configs/sam2.1/sam2.1_hiera_t.yaml \
  --output-dir outputs/ocid_full/sam2_tiny_point \
  --prompt-type point \
  --device cuda
```

```bash
PYTHONPATH=src python3 scripts/foundation_models/evaluate_sam2_auto.py \
  --index-csv outputs/ocid_full/indexes/ocid_full_objects_filtered_with_masks.csv \
  --sam2-repo /path/to/sam2 \
  --checkpoint checkpoints/sam2.1_hiera_tiny.pt \
  --config configs/sam2.1/sam2.1_hiera_t.yaml \
  --output-dir outputs/ocid_full/sam2_tiny_auto \
  --device cuda
```

EfficientSAM-Ti box:

```bash
PYTHONPATH=src python3 scripts/foundation_models/evaluate_efficientsam_box.py \
  --index-csv outputs/ocid_full/indexes/ocid_full_objects_filtered_with_masks.csv \
  --efficientsam-repo /path/to/efficientvit \
  --output-dir outputs/ocid_full/efficientsam_ti_box \
  --model-size ti \
  --device auto
```

Replace `/path/to/sam2` and `/path/to/efficientvit` with your local external
repository paths.

## 5. Generate the compatible report

After index creation only:

```bash
PYTHONPATH=src python3 scripts/analysis/summarize_ocid_massive_benchmark.py
```

After model result CSVs exist:

```bash
PYTHONPATH=src python3 scripts/analysis/summarize_ocid_massive_benchmark.py \
  --results outputs/ocid_full/results/sam_vit_b_box.csv \
            outputs/ocid_full/results/sam_vit_b_point.csv \
            outputs/ocid_full/results/sam_vit_b_auto.csv \
            outputs/ocid_full/results/fastsam_s_box.csv \
            outputs/ocid_full/results/mobilesam_box.csv \
            outputs/ocid_full/fastsam_s_point/fastsam_s_point_per_instance.csv \
            outputs/ocid_full/fastsam_s_auto/fastsam_s_auto_per_instance.csv \
            outputs/ocid_full/sam2_tiny_box/sam2_1-tiny_box_per_instance.csv \
            outputs/ocid_full/sam2_tiny_point/sam2_1-tiny_point_per_instance.csv \
            outputs/ocid_full/sam2_tiny_auto/sam2_1-tiny_auto_per_instance.csv \
            outputs/ocid_full/efficientsam_ti_box/efficientsam-ti_box_per_instance.csv
```

Strict report checking:

```bash
PYTHONPATH=src python3 scripts/analysis/summarize_ocid_massive_benchmark.py \
  --strict-results \
  --debug \
  --results outputs/ocid_full/results/sam_vit_b_box.csv \
            outputs/ocid_full/results/sam_vit_b_point.csv \
            outputs/ocid_full/results/sam_vit_b_auto.csv
```

Report outputs:

```text
docs/ocid_massive_benchmark_report.md
outputs/ocid_full/tables/
```

`outputs/ocid_full/` is generated local evidence and is ignored by Git.
