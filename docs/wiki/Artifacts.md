# Artifacts

This page lists the main files needed to inspect or reproduce the benchmark
summary.

> **Storage limitation:** The complete `results/` folder could not be included
> in Git because the raw prediction files and checkpoints are too large. The
> committed `outputs/` folder contains compact derived evidence; full raw
> results remain on the benchmark machine/AWS storage.

## Reports

| File | Purpose |
|---|---|
| [../../REPORT.md](../../REPORT.md) | Final research report. |
| [../../README.md](../../README.md) | Technical GitHub guide. |
| [../../report/](../../report/) | Supporting report pages. |
| [../../report/figures_and_tables.md](../../report/figures_and_tables.md) | Shared evidence catalog. |

## Final Figures

| Figure | Path |
|---|---|
| Dataset examples | [../../outputs/final_benchmark_assets/plots/dataset_examples.png](../../outputs/final_benchmark_assets/plots/dataset_examples.png) |
| Zero-shot heatmap | [../../outputs/final_benchmark_assets/plots/zero_shot_miou_heatmap.png](../../outputs/final_benchmark_assets/plots/zero_shot_miou_heatmap.png) |
| Prompt winners | [../../outputs/final_benchmark_assets/plots/zero_shot_dataset_prompt_winners.png](../../outputs/final_benchmark_assets/plots/zero_shot_dataset_prompt_winners.png) |
| Baseline bars | [../../outputs/final_benchmark_assets/plots/baseline_miou_bars.png](../../outputs/final_benchmark_assets/plots/baseline_miou_bars.png) |
| Speed-quality scatter | [../../outputs/final_benchmark_assets/plots/cuda_speed_quality_scatter.png](../../outputs/final_benchmark_assets/plots/cuda_speed_quality_scatter.png) |
| Lightweight trade-off | [../../outputs/final_benchmark_assets/plots/lightweight_sam_tradeoff_cuda.png](../../outputs/final_benchmark_assets/plots/lightweight_sam_tradeoff_cuda.png) |
| Challenge groups | [../../outputs/final_benchmark_assets/plots/challenge_group_weighted_iou.png](../../outputs/final_benchmark_assets/plots/challenge_group_weighted_iou.png) |

## Final Tables

| Table | Path |
|---|---|
| Best zero-shot by dataset/prompt | [../../outputs/final_benchmark_assets/tables/best_zero_shot_by_dataset_prompt.csv](../../outputs/final_benchmark_assets/tables/best_zero_shot_by_dataset_prompt.csv) |
| Best CUDA quality | [../../outputs/final_benchmark_assets/tables/best_cuda_quality_by_dataset.csv](../../outputs/final_benchmark_assets/tables/best_cuda_quality_by_dataset.csv) |
| Best CUDA speed-quality trade-off | [../../outputs/final_benchmark_assets/tables/best_cuda_tradeoff_by_dataset.csv](../../outputs/final_benchmark_assets/tables/best_cuda_tradeoff_by_dataset.csv) |
| Best lightweight CUDA trade-off | [../../outputs/final_benchmark_assets/tables/best_lightweight_cuda_tradeoff.csv](../../outputs/final_benchmark_assets/tables/best_lightweight_cuda_tradeoff.csv) |

Raw prediction files and checkpoints are stored under `results/` on the
benchmark machine. The folder could not be committed to Git because it is too
large and must be transferred separately when raw-result inspection is needed.
