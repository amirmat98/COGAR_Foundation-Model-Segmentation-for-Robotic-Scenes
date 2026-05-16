# SAM ViT-B Box-Prompt Evaluation on COGAR-SimRobotics-500

## Benchmark Versions

| Version | Objects | Mean IoU | Median IoU | Notes |
|---|---:|---:|---:|---|
| Full non-table benchmark | 7726 | 0.8750 | 0.9379 | Includes all generated non-table visible instances |
| Clean benchmark | 7274 | 0.8914 | 0.9427 | Removes degenerate, tiny, and sparse annotations |

## Clean Filtering Criteria

The clean benchmark keeps only objects satisfying:

- area >= 100 pixels
- bbox_w >= 5 pixels
- bbox_h >= 5 pixels
- visible_ratio >= 0.05

where:

```text
visible_ratio = ground-truth mask area / bounding-box area
