# Failure Mode Analysis

## Method

Failure cases were selected from the lowest-IoU objects for each completed full-dataset model:

- SAM ViT-B box
- MobileSAM box
- FastSAM-S box

For each model, the 12 worst cases were visualized. Each visualization contains four panels:

1. RGB image
2. Ground-truth mask
3. Predicted mask
4. Overlay

In the overlay, green indicates GT-only pixels, red indicates prediction-only pixels, and yellow indicates overlap.

Generated panels are stored under:

- `outputs/figures/failure_modes/sam_vit_b_box/`
- `outputs/figures/failure_modes/mobilesam_box/`
- `outputs/figures/failure_modes/fastsam_s_box/`

## Worst-case category counts

| Category | FastSAM-S box | MobileSAM box | SAM ViT-B box |
|---|---:|---:|---:|
| box | 0 | 1 | 3 |
| cable | 0 | 2 | 2 |
| connector | 0 | 1 | 0 |
| glass_object | 1 | 0 | 0 |
| metal_part | 3 | 0 | 0 |
| robot_gripper | 4 | 8 | 5 |
| screw | 1 | 0 | 0 |
| tool | 3 | 0 | 2 |

## Worst-case challenge counts

| Challenge | FastSAM-S box | MobileSAM box | SAM ViT-B box |
|---|---:|---:|---:|
| dynamic_scene | 1 | 3 | 5 |
| partial_occlusion | 1 | 4 | 3 |
| reflective_metal | 4 | 1 | 1 |
| small_parts | 4 | 1 | 0 |
| transparent_glass | 2 | 3 | 3 |

## Main observed failure modes

### Robot grippers

Robot grippers are the most common worst-case category for SAM ViT-B and MobileSAM, and they are also frequent failures for FastSAM-S. Grippers are difficult because they contain articulated parts, holes, separated visible components, and frequent contact with nearby objects. A model may segment only one visible part, merge the gripper with an adjacent object, or select a nearby wrong structure.

### Cables and thin structures

Cables are difficult for SAM ViT-B and MobileSAM. They are thin, elongated, and often overlap with other objects or background edges. These cases can produce under-segmentation, broken masks, or masks that leak into nearby structures.

### Metal parts and tools

FastSAM-S has several worst cases on metal parts and tools. These objects can be reflective, partially occluded, or visually similar to nearby objects. In FastSAM-S, failure can occur either because the all-instance proposal stage misses the correct object or because box-style selection chooses a nearby candidate mask.

### Small parts

FastSAM-S shows several worst cases under the `small_parts` challenge. Small objects such as screws and connectors occupy few pixels, making them sensitive to proposal quality and boundary errors.

### Transparent and reflective objects

Transparent glass and reflective metal produce weak or misleading visual boundaries. Transparent objects can blend with the background, while reflective objects can create highlights that look like object boundaries. These effects explain why transparent and reflective scenes appear in the worst-case set.

### Partial occlusion and dynamic scenes

Partial occlusions create incomplete visible evidence and can cause masks to leak into the occluding object. Dynamic-scene cases are also challenging because moving objects and grippers often create cluttered, overlapping configurations.

## Model-specific observations

### SAM ViT-B box

SAM ViT-B is the strongest quantitative model overall, but its worst cases still include grippers, boxes, cables, and tools. Its hardest challenge groups in the worst-case set are dynamic scenes, transparent glass, and partial occlusion.

### MobileSAM box

MobileSAM has strong overall performance for a lightweight model, but its worst cases are concentrated heavily on robot grippers. It also fails on cables and transparent/occluded scenes. This suggests that the lightweight encoder preserves much of SAM's behavior but loses robustness on complex articulated or thin objects.

### FastSAM-S box

FastSAM-S is the fastest model but has the most severe failure cases, including several zero-IoU masks. Its worst cases are concentrated in robot grippers, metal parts, tools, small parts, and reflective-metal scenes. This supports the quantitative result that FastSAM-S is useful as a high-speed baseline but is less reliable for accurate robotic object masks.

## Conclusion

The qualitative failure analysis supports the quantitative benchmark. SAM ViT-B is the most reliable model, MobileSAM provides the best lightweight accuracy trade-off, and FastSAM-S provides the strongest speed but the weakest segmentation accuracy.

The most difficult robotic-scene cases are articulated grippers, thin cables, small parts, reflective metal, transparent glass, dynamic clutter, and partial occlusion.
