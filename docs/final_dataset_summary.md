# Final COGAR-SimRobotics-500 Dataset Summary

## Dataset generation

- Simulation tool: BlenderProc
- Candidate images generated: 650
- Final clean images after audit/filtering: 500
- Final object instances: 4,471
- Bad images after final filtering: 0
- Annotation format: COCO-derived instance annotations plus exported binary masks
- Final benchmark index: `data/cogar_sim_500_final/annotations/sim_robotic_scenes_index_final_filtered.csv`

## Visual evidence

![Representative COGAR-SimRobotics-500 scenes](/outputs/figures/final_report/dataset/sample_scene_montage.png)

*Figure: Lightweight sample-scene montage showing representative challenge families in the final dataset.*

![Category count chart](/outputs/figures/final_report/dataset/category_counts.png)

*Figure: Object-instance counts by category from the final filtered benchmark index.*

![Challenge distribution chart](/outputs/figures/final_report/dataset/challenge_distribution.png)

*Figure: Object-instance counts by primary challenge group.*

## Quality statistics

- Objects per image mean: 8.94
- Objects per image median: 9
- Objects per image minimum: 4
- Objects per image maximum: 16

## Challenge distribution

| Challenge | Object instances |
|---|---:|
| small_parts | 1,269 |
| partial_occlusion | 920 |
| dynamic_scene | 797 |
| reflective_metal | 743 |
| transparent_glass | 742 |

## Category distribution

| Category | Object instances |
|---|---:|
| robot_gripper | 1,042 |
| plastic_object | 627 |
| metal_part | 555 |
| connector | 531 |
| screw | 427 |
| glass_object | 360 |
| box | 352 |
| tool | 296 |
| cable | 281 |

## Object-level challenge flags

- Reflective objects: 851
- Transparent objects: 360
- Small-part objects: 958
- Occluded-scene objects: 3,187
- Dynamic-scene objects: 797

## Filtering

The generator produced 650 candidate images. After normalization, mask export, index finalization, validation, and audit, 511 images remained in the finalized index. The audit identified 11 bad images, which were removed. The final filtered dataset contains exactly 500 clean images and 4,471 benchmark object instances.
