# COGAR-SimRobotics-500 SAM ViT-B Benchmark Results

## Dataset

This report summarizes the SAM ViT-B segmentation benchmark on **COGAR-SimRobotics-500**, a synthetic robotic-scene dataset designed for zero-shot foundation-model segmentation evaluation.

Dataset structure:

```text
data/cogar_sim_500/
├── rgb/
├── annotations/instances_all.json
├── metadata/frame_index.csv
├── metadata/categories.json
└── splits/
```

Dataset statistics:

| Item | Value |
|---|---:|
| Images | 500 |
| COCO annotations | 8570 |
| Categories | 10 |
| Clean non-table object instances | 7274 |

Categories:

```text
table
robot_gripper
metal_part
glass_object
plastic_object
screw
connector
cable
tool
box
```

Primary robotic-scene challenges:

```text
reflective_metal
transparent_glass
partial_occlusion
small_parts
dynamic_scene
```

## Clean Benchmark Filtering

The clean benchmark excludes degenerate, extremely tiny, or nearly invisible annotations.

Filtering rule:

```text
area >= 100
bbox_w >= 5
bbox_h >= 5
visible_ratio >= 0.05
```

The resulting clean benchmark contains:

```text
7274 non-table object instances
```

## Evaluated SAM ViT-B Modes

This report compares three zero-shot SAM ViT-B modes:

1. **Box prompt**
   - SAM receives a ground-truth object bounding box.
   - This represents a realistic setting where an object detector or proposal module provides object boxes.

2. **Single positive point prompt**
   - SAM receives one foreground point sampled from the ground-truth object mask.
   - This represents a lightweight interaction or sparse robotic cue.

3. **Automatic mask generation**
   - SAM receives the full RGB image and generates a set of candidate masks without object-level prompts.
   - Each ground-truth object is matched to the generated candidate mask with the highest IoU.
   - This is an oracle proposal-matching evaluation: it measures whether automatic mask generation produced a usable object candidate anywhere in its output.

## Overall Results

| SAM ViT-B mode | Objects | Mean IoU | Median IoU | Mean SAM score |
|---|---:|---:|---:|---:|
| Box prompt | 7274 | 0.8914 | 0.9427 | 0.9523 |
| Single positive point prompt | 7274 | 0.8040 | 0.9126 | 0.8784 |
| Automatic mask generation | 7274 | 0.6223 | 0.8703 | N/A |

The final ranking is:

```text
box prompt > single positive point prompt > automatic mask generation
```

## Main Result

SAM ViT-B performs best with **box prompts**.

The mean IoU drops:

| Comparison | Mean IoU drop |
|---|---:|
| Point prompt minus box prompt | -0.0874 |
| Automatic mask generation minus box prompt | -0.2690 |
| Automatic mask generation minus point prompt | -0.1817 |

Main conclusion:

```text
SAM ViT-B performs better with object-level prompts than with automatic mask generation on COGAR-SimRobotics-500.
Box prompts are the strongest mode, single foreground point prompts remain useful, and automatic mask generation is the weakest baseline.
```

## Box Prompt Summary

SAM ViT-B box prompt result:

| Metric | Value |
|---|---:|
| Objects | 7274 |
| Mean IoU | 0.8914 |
| Median IoU | 0.9427 |
| Mean SAM score | 0.9523 |

Interpretation:

```text
Box prompting gives SAM strong spatial constraints around each object.
This makes it robust for most robotic-scene objects, including reflective metal, transparent glass, and partially occluded objects.
```

## Point Prompt Summary

SAM ViT-B single positive point prompt result:

| Metric | Value |
|---|---:|
| Objects | 7274 |
| Mean IoU | 0.8040 |
| Median IoU | 0.9126 |
| Mean SAM score | 0.8784 |

Interpretation:

```text
Single positive point prompting is weaker than box prompting because one point gives less information about object extent.
The drop is most visible for ambiguous, thin, transparent, reflective, or cluttered objects.
```

## Automatic Mask Generation Summary

SAM ViT-B automatic mask generation was evaluated with:

```text
points_per_side = 16
device = cuda
model_type = vit_b
```

Automatic mask generation result:

| Metric | Value |
|---|---:|
| Objects | 7274 |
| Images with clean non-table objects | 407 |
| Mean IoU | 0.6223 |
| Median IoU | 0.8703 |
| Mean generated masks per image | 24.42 |
| Median generated masks per image | 24.00 |

Interpretation:

```text
Automatic mask generation can discover many large and visually clear objects, but it is much less reliable for small robotic components.
It often fails to generate useful masks for screws and thin parts, even when prompted SAM performs well on the same objects.
```

## Box vs Point Prompt Comparison

The earlier prompt comparison showed:

| Mode | Objects | Mean IoU | Median IoU | Mean SAM score |
|---|---:|---:|---:|---:|
| Box prompt | 7274 | 0.8914 | 0.9427 | 0.9523 |
| Single positive point prompt | 7274 | 0.8040 | 0.9126 | 0.8784 |

Conclusion:

```text
Point prompts are useful but less stable than box prompts.
The mean IoU drops from 0.8914 to 0.8040 when using a single foreground point instead of a bounding box.
```

## Automatic Mask Generation Category Failures

The weakest automatic-mask categories are:

| Category | Objects | Auto Mean IoU | Auto Median IoU | Box Mean IoU | Drop vs Box |
|---|---:|---:|---:|---:|---:|
| screw | 1243 | 0.3036 | 0.0007 | 0.8946 | -0.5910 |
| cable | 571 | 0.4993 | 0.6077 | 0.8174 | -0.3181 |
| glass_object | 771 | 0.5662 | 0.6459 | 0.8383 | -0.2722 |
| robot_gripper | 769 | 0.5972 | 0.7459 | 0.8449 | -0.2477 |
| tool | 547 | 0.5994 | 0.7363 | 0.8384 | -0.2389 |
| connector | 981 | 0.7065 | 0.9433 | 0.9253 | -0.2188 |
| metal_part | 737 | 0.7651 | 0.9559 | 0.9265 | -0.1613 |
| box | 652 | 0.8333 | 0.9748 | 0.9327 | -0.0994 |
| plastic_object | 1003 | 0.8378 | 0.9701 | 0.9489 | -0.1111 |

The most severe failure is:

```text
screw
```

The screw category has:

```text
Mean IoU:   0.3036
Median IoU: 0.0007
Drop vs box prompt: -0.5910
```

This means SAM automatic mask generation often does not generate a usable screw mask at all.

## Automatic Mask Generation Challenge Failures

The hardest challenge group is:

```text
small_parts
```

Challenge-level automatic-mask results:

| Challenge | Objects | Auto Mean IoU | Auto Median IoU | Box Mean IoU | Drop vs Box |
|---|---:|---:|---:|---:|---:|
| small_parts | 2186 | 0.5314 | 0.7529 | 0.8992 | -0.3678 |
| transparent_glass | 1163 | 0.6126 | 0.8323 | 0.8630 | -0.2504 |
| partial_occlusion | 1368 | 0.6443 | 0.8698 | 0.8837 | -0.2394 |
| reflective_metal | 1271 | 0.6669 | 0.9103 | 0.8968 | -0.2299 |
| dynamic_scene | 1286 | 0.7182 | 0.9450 | 0.9065 | -0.1882 |

Interpretation:

```text
Automatic mask generation is weakest for small robotic components.
It is also less reliable for transparent glass, partial occlusion, and reflective metal compared with prompted SAM.
```

## Hardest Categories

Across prompt modes, the hardest categories are:

```text
glass_object
robot_gripper
tool
cable
screw
```

For automatic mask generation, the hardest category is clearly:

```text
screw
```

## Hardest Challenges

Across the benchmark, the hardest challenges are:

```text
transparent_glass
partial_occlusion
reflective_metal
small_parts
```

For automatic mask generation, the strongest failure mode is:

```text
small_parts
```

## Figures

The report package includes presentation-ready figures:

```text
docs/results/cogar_sim_500/figures/
├── sam_box_vs_point_overall.png
├── sam_box_vs_point_by_category.png
├── sam_box_vs_point_by_challenge.png
├── sam_auto_by_category.png
└── sam_auto_by_challenge.png
```

Optional presentation variants may also be included:

```text
docs/results/cogar_sim_500/figures/
├── sam_box_vs_point_overall_presentation.png
├── sam_box_vs_point_by_category_presentation.png
└── sam_box_vs_point_by_challenge_presentation.png
```

## Clean CSV Tables

The curated report package includes:

```text
docs/results/cogar_sim_500/tables/
├── sam_box_clean_results.csv
├── sam_point_clean_results.csv
├── sam_auto_clean_results.csv
├── sam_box_summary.csv
├── sam_point_summary.csv
├── sam_auto_overall.csv
├── sam_box_by_category.csv
├── sam_box_by_challenge.csv
├── sam_auto_by_category.csv
├── sam_auto_by_challenge.csv
├── sam_box_vs_point_overall.csv
├── sam_box_vs_point_by_category.csv
├── sam_box_vs_point_by_challenge.csv
└── sam_box_point_auto_overall.csv
```

## Practical Recommendation

For robotic-scene segmentation with SAM ViT-B:

```text
Use box prompts when object proposals or detectors are available.
Use point prompts when lightweight human/robot interaction or sparse object cues are available.
Do not rely on automatic mask generation alone for small robotic parts.
```

Recommended pipeline:

```text
object proposal / detector -> box prompt -> SAM mask -> robotic perception module
```

Alternative lightweight interaction pipeline:

```text
foreground point -> SAM mask -> robotic perception module
```

Automatic mask generation should be treated as:

```text
zero-click object proposal baseline
```

not as a fully reliable replacement for object-level prompting in robotic scenes.

## Final Conclusion

SAM ViT-B is effective on COGAR-SimRobotics-500, but its performance depends strongly on the prompting mode.

The strongest configuration is:

```text
SAM ViT-B + box prompt
```

The full benchmark ranking is:

```text
box prompt > single positive point prompt > automatic mask generation
```

The key failure modes are:

```text
small screws
thin cables
transparent glass
robot grippers
partial occlusions
reflective metal surfaces
```

The main project conclusion is:

```text
Foundation segmentation models are promising for robotic perception, but robust robotic-scene deployment requires reliable object proposals or prompting strategies.
Automatic mask generation alone is not sufficient for small parts and cluttered robotic manipulation scenes.
```