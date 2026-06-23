# Figures and Tables Catalog

This file is the shared visual/evidence catalog for the report package. The
main report embeds the most important figures. Supporting report files should
link here instead of repeating the same figures many times.

## Core Figures

| ID | Figure | File | Primary use |
|---|---|---|---|
| F1 | Dataset examples | [`dataset_examples.png`](../outputs/final_benchmark_assets/plots/dataset_examples.png) | Shows the simulation and real clutter domains used in the benchmark. |
| F2 | Zero-shot mIoU heatmap | [`zero_shot_miou_heatmap.png`](../outputs/final_benchmark_assets/plots/zero_shot_miou_heatmap.png) | Compares zero-shot quality across models, prompts, and datasets. |
| F3 | Zero-shot dataset/prompt winners | [`zero_shot_dataset_prompt_winners.png`](../outputs/final_benchmark_assets/plots/zero_shot_dataset_prompt_winners.png) | Shows that the best zero-shot model changes by dataset and prompt. |
| F4 | Baseline mIoU bars | [`baseline_miou_bars.png`](../outputs/final_benchmark_assets/plots/baseline_miou_bars.png) | Summarizes supervised baseline quality. |
| F5 | CUDA speed-quality scatter | [`cuda_speed_quality_scatter.png`](../outputs/final_benchmark_assets/plots/cuda_speed_quality_scatter.png) | Shows the deployment trade-off between quality and FPS. |
| F6 | Lightweight SAM trade-off | [`lightweight_sam_tradeoff_cuda.png`](../outputs/final_benchmark_assets/plots/lightweight_sam_tradeoff_cuda.png) | Shows MobileSAM/EfficientSAM speed-quality behavior. |
| F7 | Challenge-group weighted IoU | [`challenge_group_weighted_iou.png`](../outputs/final_benchmark_assets/plots/challenge_group_weighted_iou.png) | Shows robotic failure modes by challenge group. |

## Core Tables

| ID | Table | File | Primary use |
|---|---|---|---|
| T1 | Best zero-shot by dataset and prompt | [`best_zero_shot_by_dataset_prompt.csv`](../outputs/final_benchmark_assets/tables/best_zero_shot_by_dataset_prompt.csv) | Prompt/model comparison. |
| T2 | Best CUDA quality by dataset | [`best_cuda_quality_by_dataset.csv`](../outputs/final_benchmark_assets/tables/best_cuda_quality_by_dataset.csv) | Highest-quality model selection. |
| T3 | Best CUDA trade-off by dataset | [`best_cuda_tradeoff_by_dataset.csv`](../outputs/final_benchmark_assets/tables/best_cuda_tradeoff_by_dataset.csv) | Real-time speed-quality recommendation. |
| T4 | Best lightweight CUDA trade-off | [`best_lightweight_cuda_tradeoff.csv`](../outputs/final_benchmark_assets/tables/best_lightweight_cuda_tradeoff.csv) | Edge/lightweight recommendation. |
| T5 | Challenge-group summary | [`challenge_group_summary.csv`](../outputs/task8_failure_analysis/challenge_group_summary.csv) | Robustness and failure-mode analysis. |
| T6 | Representative failures | [`representative_failures.csv`](../outputs/task8_failure_analysis/representative_failures.csv) | Qualitative failure selection. |

## Representative Failure Figures

| ID | File | Failure type |
|---|---|---|
| E1 | [`01_01_isaac_official_unitree_g1_fastsam_x_point_screw_iou_0.000.png`](../outputs/task8_failure_analysis/figures/01_01_isaac_official_unitree_g1_fastsam_x_point_screw_iou_0.000.png) | Isaac G1 small screw failure. |
| E2 | [`01_02_isaac_official_unitree_g1_fastsam_x_point_cable_iou_0.000.png`](../outputs/task8_failure_analysis/figures/01_02_isaac_official_unitree_g1_fastsam_x_point_cable_iou_0.000.png) | Isaac G1 thin cable failure. |
| E3 | [`02_01_isaac_official_unitree_g1_sam_vit_b_automatic_screw_iou_0.000.png`](../outputs/task8_failure_analysis/figures/02_01_isaac_official_unitree_g1_sam_vit_b_automatic_screw_iou_0.000.png) | Automatic small-part failure. |
| E4 | [`02_02_isaac_official_unitree_g1_sam_vit_b_automatic_robot_iou_0.000.png`](../outputs/task8_failure_analysis/figures/02_02_isaac_official_unitree_g1_sam_vit_b_automatic_robot_iou_0.000.png) | Robot-body confusion. |
| E5 | [`04_02_blenderproc_cogar_sim_sam_vit_h_automatic_glass_object_iou_0.000.png`](../outputs/task8_failure_analysis/figures/04_02_blenderproc_cogar_sim_sam_vit_h_automatic_glass_object_iou_0.000.png) | Transparent glass-object failure. |
| E6 | [`05_01_ocid_fastsam_x_automatic_object_iou_0.000.png`](../outputs/task8_failure_analysis/figures/05_01_ocid_fastsam_x_automatic_object_iou_0.000.png) | Real clutter/domain-gap failure. |

## Usage Rule

Use `REPORT.md` for the final narrative and this catalog for reusable visual
references. Avoid embedding the same figure repeatedly across support files.
