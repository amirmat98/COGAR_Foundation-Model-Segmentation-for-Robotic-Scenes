# Task 9: Lightweight SAM Edge-Deployment Trade-Off

This report compares MobileSAM and EfficientSAM variants against the heavier Task 4 models and the supervised Task 5 baselines where matching quality/speed rows are available.

EfficientSAM automatic mode is evaluated as grid-prompt automatic proposal generation, because the official EfficientSAM API exposes direct point/box tensor inference rather than the same automatic-mask-generator class used by SAM and MobileSAM.

## Output Tables

- `lightweight_quality.csv`: Task 9 quality metrics only.
- `speed_quality_tradeoff.csv`: quality joined with GPU/CPU speed and checkpoint size.
- `recommendations.csv`: best lightweight model per dataset, prompt mode, and device by mIoU and by mIoU-FPS product.

## Figure

![Lightweight SAM CUDA speed-quality trade-off](../../final_benchmark_assets/plots/lightweight_sam_tradeoff_cuda.png)

## Compact Summary

- Joined lightweight rows: 54
- Lightweight rows at or above 30 FPS: 0

## Recommended Lightweight Choices

| Dataset | Prompt | Device | Metric | Model | mIoU | FPS | Checkpoint MB |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| blenderproc_cogar_sim | automatic | cpu | mIoU | mobile_sam_vit_t | 0.8101 | 0.01 | 40.73 |
| blenderproc_cogar_sim | automatic | cuda | mIoU | mobile_sam_vit_t | 0.8101 | 0.20 | 40.73 |
| blenderproc_cogar_sim | box | cpu | mIoU | efficient_sam_s | 0.9098 | 0.09 | 105.74 |
| blenderproc_cogar_sim | box | cuda | mIoU | efficient_sam_s | 0.9098 | 4.31 | 105.74 |
| blenderproc_cogar_sim | point | cpu | mIoU | efficient_sam_s | 0.7956 | 0.09 | 105.74 |
| blenderproc_cogar_sim | point | cuda | mIoU | efficient_sam_s | 0.7956 | 4.31 | 105.74 |
| isaac_official_unitree_g1 | automatic | cpu | mIoU | mobile_sam_vit_t | 0.4208 | 0.01 | 40.73 |
| isaac_official_unitree_g1 | automatic | cuda | mIoU | mobile_sam_vit_t | 0.4208 | 0.22 | 40.73 |
| isaac_official_unitree_g1 | box | cpu | mIoU | efficient_sam_s | 0.7322 | 0.10 | 105.74 |
| isaac_official_unitree_g1 | box | cuda | mIoU | efficient_sam_s | 0.7322 | 4.21 | 105.74 |
| isaac_official_unitree_g1 | point | cpu | mIoU | efficient_sam_ti | 0.6031 | 0.20 | 40.98 |
| isaac_official_unitree_g1 | point | cuda | mIoU | efficient_sam_ti | 0.6031 | 8.14 | 40.98 |
| ocid | automatic | cpu | mIoU | mobile_sam_vit_t | 0.8105 | 0.01 | 40.73 |
| ocid | automatic | cuda | mIoU | mobile_sam_vit_t | 0.8105 | 0.23 | 40.73 |
| ocid | box | cpu | mIoU | efficient_sam_s | 0.8545 | 0.10 | 105.74 |
| ocid | box | cuda | mIoU | efficient_sam_s | 0.8545 | 4.29 | 105.74 |
| ocid | point | cpu | mIoU | mobile_sam_vit_t | 0.6739 | 0.63 | 40.73 |
| ocid | point | cuda | mIoU | mobile_sam_vit_t | 0.6739 | 15.56 | 40.73 |
| blenderproc_cogar_sim | automatic | cpu | miou_fps_product | efficient_sam_ti | 0.1205 | 0.08 | 40.98 |
| blenderproc_cogar_sim | automatic | cuda | miou_fps_product | efficient_sam_ti | 0.1205 | 2.52 | 40.98 |
| blenderproc_cogar_sim | box | cpu | miou_fps_product | mobile_sam_vit_t | 0.8827 | 0.65 | 40.73 |
| blenderproc_cogar_sim | box | cuda | miou_fps_product | mobile_sam_vit_t | 0.8827 | 15.74 | 40.73 |
| blenderproc_cogar_sim | point | cpu | miou_fps_product | mobile_sam_vit_t | 0.7399 | 0.68 | 40.73 |
| blenderproc_cogar_sim | point | cuda | miou_fps_product | mobile_sam_vit_t | 0.7399 | 15.62 | 40.73 |
| isaac_official_unitree_g1 | automatic | cpu | miou_fps_product | mobile_sam_vit_t | 0.4208 | 0.01 | 40.73 |
| isaac_official_unitree_g1 | automatic | cuda | miou_fps_product | mobile_sam_vit_t | 0.4208 | 0.22 | 40.73 |
| isaac_official_unitree_g1 | box | cpu | miou_fps_product | mobile_sam_vit_t | 0.6926 | 0.58 | 40.73 |
| isaac_official_unitree_g1 | box | cuda | miou_fps_product | mobile_sam_vit_t | 0.6926 | 16.89 | 40.73 |
| isaac_official_unitree_g1 | point | cpu | miou_fps_product | mobile_sam_vit_t | 0.5164 | 0.57 | 40.73 |
| isaac_official_unitree_g1 | point | cuda | miou_fps_product | mobile_sam_vit_t | 0.5164 | 16.84 | 40.73 |
| ocid | automatic | cpu | miou_fps_product | efficient_sam_ti | 0.1027 | 0.08 | 40.98 |
| ocid | automatic | cuda | miou_fps_product | efficient_sam_s | 0.1406 | 1.93 | 105.74 |
| ocid | box | cpu | miou_fps_product | mobile_sam_vit_t | 0.8238 | 0.64 | 40.73 |
| ocid | box | cuda | miou_fps_product | mobile_sam_vit_t | 0.8238 | 15.52 | 40.73 |
| ocid | point | cpu | miou_fps_product | mobile_sam_vit_t | 0.6739 | 0.63 | 40.73 |
| ocid | point | cuda | miou_fps_product | mobile_sam_vit_t | 0.6739 | 15.56 | 40.73 |
