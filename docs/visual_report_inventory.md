# Visual Report Inventory

This inventory records the existing visual and table assets used to upgrade the
final assignment reports with charts, sample scenes, and failure evidence.

The full raw inventory files were generated locally with:

```bash
find outputs -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" -o -iname "*.svg" \) | sort > /tmp/cogar_existing_figures.txt
find outputs -type f \( -iname "*.csv" -o -iname "*.json" \) | sort > /tmp/cogar_existing_tables.txt
find data -type f \( -iname "*.csv" -o -iname "*.json" -o -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | sort > /tmp/cogar_existing_data_assets.txt
```

## Inventory summary

| Asset class | Count | Notes |
|---|---:|---|
| Existing output figures | 51,310 | Mostly generated masks and visualizations; only lightweight representative panels are reused in reports |
| Existing output CSV/JSON tables | 237 | Includes final model summaries, per-category tables, dataset audits, and baseline summaries |
| Existing data CSV/JSON/image assets | 8,761 | Includes local dataset indexes, masks, and RGB images; thumbnails only are copied into final report figures |

## Report-relevant existing figures

| Existing visual asset | Use in reports |
|---|---|
| `outputs/figures/failure_modes/sam_vit_b_box/*.png` | Task 8, `failure_mode_analysis.md`, README failure montage |
| `outputs/figures/failure_modes/mobilesam_box/*.png` | Task 8 and lightweight failure examples |
| `outputs/figures/failure_modes/fastsam_s_box/*.png` | Task 8 and speed-first failure examples |
| `outputs/cogar_sim_500/analysis_prompt_comparison/figures_presentation/*.png` | Historical prompt-comparison reference; superseded by final generated charts |
| `outputs/cogar_sim_500/analysis_sam_auto_masks/figures/*.png` | SAM automatic-mask reference; superseded by final generated charts |
| `data/cogar_sim_500_final/rgb/*.png` | Source for lightweight sample-scene thumbnails and montage |

## Report-relevant existing tables

| Existing table | Use in generated report figures |
|---|---|
| `outputs/tables/dataset_audit_final_filtered/category_counts.csv` | Dataset category count chart |
| `outputs/tables/dataset_audit_final_filtered/challenge_counts.csv` | Challenge distribution chart |
| `outputs/tables/sim_sam_vit_b_final_prompt_summary.csv` | SAM ViT-B prompt-mode charts |
| `outputs/tables/sam2/final_box_cuda/overall_summary.csv` | SAM2.1-Tiny box charts |
| `outputs/tables/sam2/final_point_cuda/overall_summary.csv` | SAM2.1-Tiny point charts |
| `outputs/tables/sam2/final_auto_cuda/overall_summary.csv` | SAM2.1-Tiny auto charts |
| `outputs/tables/fastsam/point_final/overall_summary.csv` | FastSAM-S point charts |
| `outputs/tables/fastsam/auto_final/overall_summary.csv` | FastSAM-S auto charts |
| `outputs/tables/final_box_prompt_model_summary.csv` | SAM ViT-B, MobileSAM, and FastSAM-S box comparison |
| `outputs/tables/efficientsam/final_ti_cuda_fixed/overall_summary.csv` | EfficientSAM-Ti charts |
| `outputs/tables/sim_sam_vit_h_*_summary.csv` | SAM ViT-H CPU subset charts |
| `outputs/tables/sam2/final_box_cuda/mean_iou_by_category.csv` | Per-category IoU chart |
| `outputs/tables/maskrcnn_resnet50_fpn_cogar_full_per_class.csv` | Per-category IoU chart |
| `outputs/tables/maskrcnn_resnet50_fpn_cogar_full_summary.csv` | Supervised baseline summary |
| `outputs/tables/failure_modes/*.csv` | Failure-mode count tables already included in failure reports |

## Generated final report figures

| Generated figure | Inserted into reports |
|---|---|
| `/outputs/figures/final_report/dataset/sample_scene_montage.png` | README, Task 1, Task 2, Task 3, final dataset summary |
| `/outputs/figures/final_report/dataset/category_counts.png` | README, Task 1, Task 6, final dataset summary |
| `/outputs/figures/final_report/dataset/challenge_distribution.png` | README, Task 1, final dataset summary |
| `/outputs/figures/final_report/dataset/simulation_pipeline.png` | Task 2 |
| `/outputs/figures/final_report/metrics/mean_iou_by_model_prompt.png` | README, Task 4, Task 6, model-specific result reports |
| `/outputs/figures/final_report/metrics/boundary_f1_by_model_prompt.png` | Task 4, Task 6, model-specific result reports |
| `/outputs/figures/final_report/speed/fps_comparison.png` | README, Task 7, model-specific result reports |
| `/outputs/figures/final_report/edge_tradeoff/iou_vs_fps_tradeoff.png` | README, Task 7, Task 9, recommendation guide |
| `/outputs/figures/final_report/metrics/supervised_baselines_summary.png` | Task 5, Task 6, YOLOv8n-seg report, Mask R-CNN report |
| `/outputs/figures/final_report/metrics/per_category_iou.png` | Task 6, final cross-model report, SAM2 and Mask R-CNN reports |
| `/outputs/figures/final_report/failure_modes/failure_mode_montage.png` | README, Task 8, failure-mode report |

## Missing visuals generated from existing data

| Missing visual | Source data | Generated output |
|---|---|---|
| Dataset composition chart | Dataset audit CSV | `category_counts.png` |
| Challenge distribution chart | Dataset audit CSV | `challenge_distribution.png` |
| Simulation pipeline diagram | Documented workflow | `simulation_pipeline.png` |
| Zero-shot mean IoU comparison | Final result CSVs/docs | `mean_iou_by_model_prompt.png` |
| Boundary F1 comparison | Final result CSVs/docs | `boundary_f1_by_model_prompt.png` |
| FPS comparison | Final result CSVs/docs | `fps_comparison.png` |
| Edge trade-off scatter | Final result CSVs/docs | `iou_vs_fps_tradeoff.png` |
| Supervised baseline chart | Final YOLO/Mask R-CNN docs and CSV | `supervised_baselines_summary.png` |
| Per-category IoU chart | SAM2 and Mask R-CNN category CSVs | `per_category_iou.png` |
| Failure montage | Existing failure panels | `failure_mode_montage.png` |

## Large files excluded from final report visuals

Raw generated masks, full-resolution output dumps, raw simulation exports,
checkpoints, and model weights were not added to the final report figure set.
The generated final-report figures are all below 5 MB.
