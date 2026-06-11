# OCID Massive Benchmark Report

## Dataset

- Source root: `/mnt/Info/COGAR_DATASETs/OCID-dataset`
- Image index: `outputs/ocid_full/indexes/ocid_full_images.csv`
- Object index: `outputs/ocid_full/indexes/ocid_full_objects_filtered_with_masks.csv`
- Source RGB-label pairs: 2,390
- Indexed RGB images after filtering: 2,287
- Object instances after filtering: 17,545
- Exported binary object masks: 17,545
- Sequences: 178

## Object Set Distribution

| object_set | objects |
| --- | --- |
| ARID20 | 10740 |
| ARID10 | 4219 |
| YCB10 | 2586 |

## Scene Type Distribution

| scene_type | objects |
| --- | --- |
| curved | 1729 |
| mixed | 1729 |
| cuboid | 877 |
| box | 852 |
| non-fruits | 848 |
| seq01 | 840 |
| seq03 | 840 |
| seq08 | 833 |
| seq06 | 832 |
| seq07 | 832 |
| seq05 | 829 |
| seq04 | 826 |
| seq09 | 825 |
| seq12 | 824 |
| seq10 | 820 |
| seq13 | 820 |
| seq02 | 810 |
| seq11 | 809 |
| fruits | 770 |

## Surface Distribution

| surface | objects |
| --- | --- |
| table | 8884 |
| floor | 8661 |

## Camera View Distribution

| camera_view | objects |
| --- | --- |
| top | 8846 |
| bottom | 8699 |

## Result CSV Summary

| result_csv | status | rows | mean_iou | median_iou | iou_ge_075 | iou_lt_010 | mean_boundary_f1 | mean_fps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| outputs/ocid_full/results/sam_vit_b_box.csv | available | 17545 | 0.8624231647762326 | 0.8839112963238577 | 0.9182673126246794 | 0.0005699629524080935 | 0.8691325538238913 | 169.17713095294692 |
| outputs/ocid_full/results/sam_vit_b_point.csv | available | 17545 | 0.6112237663504039 | 0.7639073729039128 | 0.5144485608435452 | 0.10800797948133371 | 0.6017144998154408 | 151.90865011918706 |
| outputs/ocid_full/results/sam_vit_b_auto_fast16.csv | available | 17545 | 0.6499972392193917 | 0.8541213768115942 | 0.6470789398689085 | 0.16756910800797947 | 0.6800130302978263 | 0.673239069363305 |

## Compatible Evaluation Commands

SAM ViT-B box prompts:

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

SAM ViT-B point prompts:

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

SAM ViT-B automatic masks:

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

Regenerate this report after result CSVs exist:

```bash
PYTHONPATH=src python3 scripts/analysis/summarize_ocid_massive_benchmark.py \
  --results outputs/ocid_full/results/sam_vit_b_box.csv \
            outputs/ocid_full/results/sam_vit_b_point.csv \
            outputs/ocid_full/results/sam_vit_b_auto_fast16.csv \
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

## Notes

- This is a real-world OCID generalization benchmark, separate from the simulated COGAR-SimRobotics-500 assignment benchmark.
- The index uses OCID instance-label images to derive object masks, boxes, and point prompts.
- Additional FastSAM, MobileSAM, SAM2, and EfficientSAM OCID commands are documented in `docs/ocid_massive_benchmark.md`.
- AWS packaging and execution commands are documented in `docs/aws_ocid_benchmark.md`.
- Full-model runs can be expensive; use `--limit` for smoke tests before launching full OCID runs.
