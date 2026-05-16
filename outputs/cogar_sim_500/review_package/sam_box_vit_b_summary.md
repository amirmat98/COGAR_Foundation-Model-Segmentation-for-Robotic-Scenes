# SAM ViT-B Box-Prompt Evaluation on COGAR-SimRobotics-500

## Setup

- Dataset: COGAR-SimRobotics-500
- Model: SAM ViT-B
- Prompt type: bounding box
- Evaluated objects: 7726
- Excluded category: table
- Metric: IoU between SAM prediction and binary ground-truth mask

## Overall Results

| Metric | Value |
|---|---:|
| Mean IoU | 0.8750 |
| Median IoU | 0.9379 |
| Mean SAM score | 0.9450 |
| Min IoU | 0.0000 |
| Max IoU | 1.0000 |

## Main Observations

SAM ViT-B performs strongly on most synthetic robotic objects when given box prompts. The median IoU is high, showing that most object instances are segmented accurately. However, the mean IoU is lower than the median because a subset of hard failures has near-zero IoU.

The most difficult categories are cable, tool, glass_object, robot_gripper, and screw. These objects are challenging because they are thin, reflective, transparent, small, or partially occluded.

The hardest primary challenge is transparent_glass, followed by partial_occlusion and small_parts. This confirms that the synthetic dataset creates meaningful stress cases for robotic-scene segmentation.

## Generated Artifacts

- Full enriched results:
  `outputs/indexes/cogar_sim_500_sam_box_no_table_results_enriched.csv`

- Category summary:
  `outputs/cogar_sim_500/analysis/sam_box_iou_by_category.csv`

- Challenge summary:
  `outputs/cogar_sim_500/analysis/sam_box_iou_by_challenge.csv`

- Failure visualizations:
  `outputs/cogar_sim_500/sam_box_failure_visualizations/`

- Figures:
  `outputs/cogar_sim_500/analysis/figures/`
