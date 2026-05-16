# Legacy OCID SAM Box-Prompt Qualitative Failure Analysis

## Experiment Summary

This legacy OCID prototype experiment evaluates the Segment Anything Model (SAM) on object-level segmentation in cluttered robotic scenes from the OCID dataset. The selected debug sequence is `YCB10/table/top/mixed/seq21`. The model used is SAM ViT-B, and each object instance is segmented using a bounding-box prompt derived from the object-level ground-truth index.

The legacy evaluation includes 52 filtered object instances. For each object, the SAM predicted mask is compared against the exported binary ground-truth mask using intersection over union (IoU).

| Item | Value |
| --- | --- |
| Dataset | Legacy OCID prototype |
| Sequence | `YCB10/table/top/mixed/seq21` |
| Model | SAM ViT-B |
| Prompt type | Bounding box |
| Evaluated object instances | 52 |
| Evaluation metric | IoU between SAM predicted mask and binary ground-truth mask |

## Quantitative Summary

| Metric | Value |
| --- | ---: |
| Evaluated objects | 52 |
| Mean IoU | 0.8495 |
| Median IoU | 0.8784 |
| Minimum IoU | 0.7087 |
| Maximum IoU | 0.9126 |
| Mean SAM score | 0.9629 |

## Best-Case Analysis

The strongest results are obtained for object ID 5. This object appears compact, visually distinct, and mostly isolated from nearby clutter. Its boundary is clear enough for the box prompt to provide a useful spatial constraint, and the prompt tightly covers the object without including much ambiguous background. As a result, SAM predicts smooth masks that overlap well with the binary ground-truth masks.

| Row | Object ID | IoU | SAM score | Relevant visualization |
| ---: | ---: | ---: | ---: | --- |
| 23 | 5 | 0.9126 | 0.9689 | `outputs/sam_box_prompt/visualizations/row_0023_object_5_sam_visualization.png` |
| 38 | 5 | 0.9094 | 0.9679 | `outputs/sam_box_prompt/visualizations/row_0038_object_5_sam_visualization.png` |
| 5 | 5 | 0.9038 | 0.9841 | `outputs/sam_box_prompt/visualizations/row_0005_object_5_sam_visualization.png` |

These examples show that SAM ViT-B with box prompts is effective when the target object has limited ambiguity, a compact shape, and a well-aligned bounding box.

## Worst-Case Analysis

| Row | Object ID | IoU | SAM score | Relevant visualization |
| ---: | ---: | ---: | ---: | --- |
| 47 | 6 | 0.7087 | 0.8605 | `outputs/sam_box_prompt/visualizations/row_0047_object_6_sam_visualization.png` |
| 19 | 7 | 0.7397 | 0.9788 | `outputs/sam_box_prompt/visualizations/row_0019_object_7_sam_visualization.png` |
| 40 | 7 | 0.7441 | 0.9786 | `outputs/sam_box_prompt/visualizations/row_0040_object_7_sam_visualization.png` |
| 25 | 7 | 0.7705 | 0.9785 | `outputs/sam_box_prompt/visualizations/row_0025_object_7_sam_visualization.png` |

Object ID 7 is a repeated failure pattern. In these cases, SAM predicts a clean and compact mask, but the predicted region is smoother and smaller than the binary ground-truth mask. The ground-truth mask is more irregular, and because the object is small, modest boundary differences cause a large relative drop in IoU. This indicates that high-quality visual masks do not always maximize agreement with pixel-level ground truth, especially for small objects.

Object ID 6 in row 47 is the lowest-IoU case. The object has a more complex shape, including internal structure, and is close to neighboring objects. SAM captures the main visible region, but it does not fully reproduce the ground-truth shape. In contrast to the object ID 7 cases, both IoU and the SAM internal score are lower, suggesting that the model is also less confident about this prediction.

## Main Failure Modes

**Boundary simplification.** SAM often produces smooth masks, while the legacy OCID binary ground-truth masks may contain irregular boundaries. This mismatch reduces IoU even when the predicted mask is visually plausible.

**Small-object sensitivity.** For small objects, a few pixels of under-segmentation or boundary displacement have a large effect on IoU. This is especially visible in repeated failures for object ID 7.

**Clutter and ambiguity.** Objects near other objects or with complex local appearance are harder to isolate from a box prompt alone. Neighboring clutter can make the intended object boundary ambiguous.

**SAM score versus ground-truth IoU.** SAM's internal score is not always aligned with the external ground-truth IoU. Several object ID 7 failures have SAM scores near 0.979 despite substantially lower IoU values, so SAM score should be treated as a model confidence signal rather than a substitute for benchmark evaluation.

## Main Conclusion

SAM ViT-B with bounding-box prompts performs well on compact, visually clear, and relatively isolated objects in the legacy OCID debug sequence. The best cases exceed 0.90 IoU and demonstrate that a tight box prompt can be an effective zero-shot segmentation cue.

Performance decreases for small irregular objects, objects close to clutter, and objects with complex internal structure. The qualitative analysis also shows that SAM's internal score should not be used alone as the evaluation metric. External comparison against binary ground-truth masks remains necessary for reliable benchmark conclusions.
