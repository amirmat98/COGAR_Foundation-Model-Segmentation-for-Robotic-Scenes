# Assignment Task 8: Qualitative Failure Mode Analysis

## Task requirement

Analyze failure modes qualitatively, identifying where and why the segmentation models fail in robotic scenarios.

## Status

Task 8 is completed.

The project includes a qualitative failure-mode analysis based on low-IoU predictions, category-level metrics, challenge tags, and visual inspection of generated failure panels.

Main supporting file:

- `docs/failure_mode_analysis.md`

Failure visualizations are stored under:

- `outputs/figures/failure_modes/sam_vit_b_box/`
- `outputs/figures/failure_modes/mobilesam_box/`
- `outputs/figures/failure_modes/fastsam_s_box/`

Each visualization contains:

1. RGB image
2. Ground-truth mask
3. Predicted mask
4. Overlay

In the overlays:

- green = ground-truth-only pixels
- red = prediction-only pixels
- yellow = overlap

## Method

Failure cases were selected from the lowest-IoU objects for completed benchmark models and then inspected qualitatively.

The analysis uses four evidence sources:

1. Lowest-IoU qualitative panels.
2. Per-category IoU tables.
3. Per-challenge IoU tables.
4. Comparison between prompt modes and model families.

The goal is not only to identify which model has lower score, but also to explain why segmentation fails in robotic scenes.

## Main failure modes

| Failure mode | Typical categories | Main cause | Typical segmentation error | Robotic impact |
|---|---|---|---|---|
| Articulated robot parts | `robot_gripper` | Multiple fingers, holes, joints, contact with objects | Partial mask, merged mask, wrong nearby structure | Bad grasp-state understanding and poor robot-object separation |
| Thin structures | `cable`, thin tool parts | Low pixel width, long curved geometry, background similarity | Broken masks or under-segmentation | Unreliable cable/tool localization |
| Small parts | `screw`, `connector`, small metal/plastic objects | Few pixels and weak boundaries | Missed object, zero-IoU mask, unstable boundary | Small-part manipulation becomes unreliable |
| Transparent objects | `glass_object` | Weak RGB boundaries and background visibility through object | Mask leaks to background or misses object interior | Poor glass-object grasping and collision reasoning |
| Reflective metal | `metal_part`, tools | Specular highlights and misleading edges | Boundary follows reflection instead of object | Incorrect part extent and pose estimation |
| Partial occlusion | many categories | Only part of object visible | Model segments visible part only or merges with occluder | Incomplete object mask for grasp planning |
| Dynamic clutter | gripper-object interaction scenes | Contact, overlap, motion-like configurations | Confusion between robot, object, and background | Poor scene understanding during manipulation |
| Prompt ambiguity | point-prompt and auto modes | Prompt may match object part, whole object, or nearby object | Part-level or wrong-instance mask | Unstable prompt-conditioned segmentation |
| Automatic-mask proposal miss | FastSAM/SAM2 auto | Correct instance not proposed or low-ranked | Best available mask is wrong or missing | Prompt-free deployment is less reliable |

## Model-specific observations

### SAM ViT-B

SAM ViT-B is one of the strongest full-dataset zero-shot models, especially with box prompts.

Its main failures occur on:

- robot grippers
- cables
- tools
- transparent or reflective objects
- occluded/dynamic scenes

Typical failure behavior:

- box prompt helps strongly, but cannot fully solve ambiguous object boundaries
- point prompt is weaker when the selected point lies on a part rather than the full object
- automatic mask generation may split objects into parts or miss small objects

Robotic interpretation:

SAM ViT-B is reliable when an external detector or planner can provide good bounding boxes, but it is less reliable as a fully automatic perception system in cluttered manipulation scenes.

### SAM2.1-Tiny

SAM2.1-Tiny performs very strongly with box and point prompts, but automatic mask generation is much weaker.

Important observed pattern:

- box prompt is strongest
- point prompt is lower, especially for grippers, cables, tools, and transparent objects
- auto mode struggles heavily with screws, small parts, robot grippers, and cables

Robotic interpretation:

SAM2.1-Tiny is a strong promptable model for robotic perception when object proposals are available. However, prompt-free automatic segmentation is not reliable enough for small objects and cluttered manipulation scenes on this dataset.

### FastSAM-S

FastSAM-S is the fastest model family in the benchmark, but its failure cases are more severe.

Typical failure behavior:

- proposal stage may miss the correct object
- box-selected candidate may correspond to the wrong nearby instance
- small parts, reflective parts, and tools can produce very low or zero IoU
- boundaries are less precise than SAM/SAM2-style models

Robotic interpretation:

FastSAM-S is useful when speed is the priority, but it is risky for precision manipulation where mask accuracy matters.

### MobileSAM and EfficientSAM-Ti

Lightweight SAM-style models provide useful speed/accuracy trade-offs but are less robust on complex shapes.

Typical failure behavior:

- less reliable on articulated grippers
- weaker on cables and thin structures
- lower robustness under occlusion and transparent/reflective effects

Robotic interpretation:

These models are useful for edge-style experiments, but they need careful prompting, higher-resolution crops, or downstream filtering for manipulation use.

### YOLOv8n-seg

YOLOv8n-seg is the most practical supervised real-time baseline.

Typical strengths:

- automatic instance segmentation
- strong image-level FPS
- good mask AP on the supervised test split

Typical limitations:

- depends on fine-tuned training data
- may generalize less well to unseen object layouts or real-world domain shift
- small, thin, transparent, and reflective objects remain challenging

Robotic interpretation:

YOLOv8n-seg is the strongest deployment-style baseline when the object categories and visual domain are known.

### Mask R-CNN ResNet-50 FPN

Mask R-CNN provides a classical supervised instance-segmentation baseline.

Observed difficult categories include:

- cable
- screw
- tool
- glass_object

Typical failure behavior:

- missed or low-quality masks for small/thin objects
- less precise boundaries under clutter
- slower image-level FPS than YOLOv8n-seg

Robotic interpretation:

Mask R-CNN is useful as a supervised comparison point, but it is not the best real-time choice on the available GTX 1050 hardware.

## Why robotic scenes are difficult

Robotic manipulation scenes are harder than clean object-centric images because they include:

1. physical contact between robot and objects
2. partial occlusion from grippers and clutter
3. small industrial parts
4. thin cables and tool structures
5. reflective and transparent materials
6. visually similar adjacent objects
7. dynamic manipulation layouts
8. ambiguous object-vs-part boundaries

These properties create segmentation failures that directly affect downstream robotic tasks such as grasp planning, collision checking, object tracking, and pose estimation.

## Prompt-mode failure interpretation

### Box prompts

Box prompts are the most reliable because they constrain the target object spatially.

Remaining failures usually happen when:

- the box contains multiple touching objects
- the object is thin or partially occluded
- the true object boundary is visually weak
- the model chooses a part instead of the full object

### Point prompts

Point prompts are more ambiguous.

Failures happen when:

- the point lies on a small visible part of a larger object
- the point is close to another object boundary
- the object contains holes or disconnected visible regions
- foreground and background have similar appearance

### Automatic mask generation

Automatic mask generation is the hardest mode.

Failures happen when:

- the correct object is never proposed
- small objects are filtered out
- nearby objects are merged
- one object is split into multiple part masks
- the selected mask has high confidence but poor object-level IoU

## Practical recommendations

For robotic deployment:

1. Use YOLOv8n-seg when real-time automatic segmentation is the main goal.
2. Use SAM2.1-Tiny or SAM ViT-B with box prompts when accuracy is more important than full automation.
3. Use FastSAM-S when very high speed is required and lower mask accuracy is acceptable.
4. Avoid relying on CPU inference for large SAM models.
5. Add depth or multi-view sensing for transparent and reflective objects.
6. Use object proposals, tracking, or robot state to reduce prompt ambiguity.
7. Use higher-resolution crops for small parts and cables.
8. Fine-tune supervised models if the deployment object categories are known.

## Conclusion

Task 8 is completed.

The main qualitative failure modes are articulated grippers, thin cables, small parts, transparent glass, reflective metal, partial occlusion, dynamic clutter, prompt ambiguity, and automatic-mask proposal errors.

These failures explain the quantitative trends observed in the benchmark: box-prompted SAM/SAM2 models achieve the best accuracy, FastSAM-S achieves the best speed but weaker reliability, YOLOv8n-seg is the strongest real-time supervised baseline, and Mask R-CNN is useful as a classical comparison but less suitable for real-time deployment on the available hardware.
