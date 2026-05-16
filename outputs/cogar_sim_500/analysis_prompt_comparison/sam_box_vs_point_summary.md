# SAM ViT-B Prompt Comparison on COGAR-SimRobotics-500

## Setup

- Dataset: COGAR-SimRobotics-500
- Benchmark subset: clean non-table object instances
- Model: SAM ViT-B
- Prompt types compared:
  - Bounding box prompt
  - Single positive foreground point prompt
- Metric: IoU between SAM prediction and binary ground-truth mask
- Objects evaluated: 7274

## Overall Results

| Prompt type | Objects | Mean IoU | Median IoU | Mean SAM score |
|---|---:|---:|---:|---:|
| Box prompt | 7274 | 0.8914 | 0.9427 | 0.9523 |
| Point prompt | 7274 | 0.8040 | 0.9126 | 0.8784 |

## Main Finding

SAM ViT-B performs better with box prompts than with single foreground point prompts on COGAR-SimRobotics-500.

The mean IoU decreases from 0.8914 to 0.8040 when replacing the box prompt with a single positive point prompt. This indicates that prompt quality has a strong effect on segmentation quality in robotic scenes.

## Category-Level Observation

The largest drops appear for:

- glass_object
- robot_gripper
- tool
- cable

These categories are difficult because they are often transparent, reflective, thin, partially occluded, or visually ambiguous.

## Challenge-Level Observation

The largest drops appear for:

- transparent_glass
- partial_occlusion
- reflective_metal

This confirms that point prompts are less reliable in visually ambiguous robotic scenes.

## Conclusion

The box-prompt baseline should be treated as the stronger SAM ViT-B prompt setting. The point-prompt result is still useful because it shows how segmentation quality changes when less spatial guidance is provided. This comparison supports the broader conclusion that foundation segmentation models are sensitive to prompt type in robotic-scene understanding.
