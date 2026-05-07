# COGAR-SimRobotics-500 SAM ViT-B Prompt Benchmark

## Dataset

This report summarizes the SAM ViT-B prompt benchmark on **COGAR-SimRobotics-500**.

- Images: 500
- COCO annotations: 8570
- Categories: 10
- Clean benchmark filter: `area >= 100, bbox_w >= 5, bbox_h >= 5, visible_ratio >= 0.05`
- Clean benchmark object instances: 7274

## Evaluated prompts

The benchmark compares two zero-shot prompt types on the same clean non-table object instances:

1. **Box prompt**: the model receives a ground-truth object bounding box.
2. **Single positive point prompt**: the model receives one foreground point sampled from the object mask.

## Main quantitative result

| Prompt | Objects | Mean IoU | Median IoU | Mean SAM score |
|---|---:|---:|---:|---:|
| Box prompt | 7274 | 0.8914 | 0.9427 | 0.9523 |
| Single positive point prompt | 7274 | 0.8040 | 0.9126 | 0.8784 |
| Point minus box | — | -0.0874 | — | -0.0740 |

The mean IoU drops from **0.8914** with box prompts to **0.8040** with single positive point prompts.
The absolute mean IoU difference is **-0.0874**.

## Interpretation

SAM ViT-B performs better and more consistently with bounding-box prompts than with single foreground point prompts on COGAR-SimRobotics-500.

The result is expected because a box prompt gives SAM stronger spatial constraints around the object extent, while a single point prompt gives weaker information in cluttered robotic scenes. This is especially important for objects with ambiguous boundaries, transparency, specular highlights, thin structures, or heavy occlusion.

## Harder categories for point prompts

The most problematic categories for point prompts compared with box prompts are:

- `glass_object`
- `robot_gripper`
- `tool`
- `cable`

## Harder challenge types

The most difficult challenge groups are:

- `transparent_glass`
- `partial_occlusion`
- `reflective_metal`

## Included clean tables

- `tables/sam_box_clean_results.csv`
- `tables/sam_point_clean_results.csv`
- `tables/sam_box_summary.csv`
- `tables/sam_point_summary.csv`
- `tables/sam_box_by_category.csv`
- `tables/sam_box_by_challenge.csv`
- `tables/sam_box_vs_point_overall.csv`
- `tables/sam_box_vs_point_by_category.csv`
- `tables/sam_box_vs_point_by_challenge.csv`

## Included figures

- `figures/sam_box_vs_point_by_category.png`
- `figures/sam_box_vs_point_by_challenge.png`
- `figures/sam_box_vs_point_overall.png`

## Report conclusion

For the current COGAR-SimRobotics-500 benchmark, **SAM ViT-B box prompting is the stronger baseline**.
Single positive point prompting remains useful, but it is less stable on transparent, reflective, occluded, and thin robotic-scene objects.
