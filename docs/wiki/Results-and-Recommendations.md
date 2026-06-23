# Results and Recommendations

The conclusions are conditional. No single model is best for all robotic
scenarios.

## Key Results

| Scenario | Best setting | Main result |
|---|---|---|
| Best synthetic quality | SAM ViT-H box on BlenderProc | mIoU 0.923, boundary F1 0.905, mask AP 0.868 |
| Best Isaac quality | SAM ViT-H box on Isaac G1 | mIoU 0.752, boundary F1 0.874, mask AP 0.678 |
| Best OCID quality | DeepLabV3+ | mIoU 0.963, boundary F1 0.880, 37.811 FPS |
| Best BlenderProc speed-quality | YOLOv8-seg | mIoU 0.861, 41.647 FPS |
| Best lightweight box trade-off | MobileSAM | 0.883 BlenderProc, 0.693 Isaac G1, 0.824 OCID |

## Recommendations

| Robotic scenario | Recommended model family |
|---|---|
| Highest mask quality | SAM ViT-H / SAM2 with box prompts |
| Prompt-guided manipulation | SAM/SAM2/MobileSAM with box prompts |
| Real-time control with labels | YOLOv8-seg or DeepLabV3+ |
| Edge-oriented prompted perception | MobileSAM or EfficientSAM |
| Open-ended object discovery | Automatic mask generation with filtering |
| Transparent, reflective, occluded, small objects | Any model plus extra validation |

Core visuals:

- [Zero-shot heatmap](../../outputs/final_benchmark_assets/plots/zero_shot_miou_heatmap.png)
- [Speed-quality scatter](../../outputs/final_benchmark_assets/plots/cuda_speed_quality_scatter.png)
- [Challenge-group plot](../../outputs/final_benchmark_assets/plots/challenge_group_weighted_iou.png)

More detail:

- [../../report/05_results_congruence_and_conclusions.md](../../report/05_results_congruence_and_conclusions.md)
- [../../REPORT.md#5-congruence-of-results-and-conclusions](../../REPORT.md#5-congruence-of-results-and-conclusions)
