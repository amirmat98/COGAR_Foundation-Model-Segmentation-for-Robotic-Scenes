# Final Mask R-CNN Baseline Results

## Role in the benchmark

Mask R-CNN was added as a supervised instance-segmentation baseline for Assignment Task 5.

Unlike the zero-shot foundation models, this baseline was fine-tuned on the COGAR simulation training split. It is therefore used as a supervised comparison point, not as a zero-shot method.

Mask R-CNN is suitable for this role because it is an instance-segmentation model: it detects objects and predicts a segmentation mask for each instance. The original Mask R-CNN paper describes it as extending Faster R-CNN with a parallel mask-prediction branch, and TorchVision provides a Mask R-CNN implementation for instance segmentation.

## Training setup

| Item | Value |
|---|---:|
| Model | Mask R-CNN ResNet-50 FPN |
| Implementation | TorchVision |
| Pretraining | COCO weights |
| Train split | full training split |
| Test split | 75 images / 679 objects |
| Epochs | 5 |
| Batch size | 1 |
| Image size | min-size 320, max-size 512 |
| Device | NVIDIA GTX 1050 4 GB |
| AMP | enabled |
| Backbone | frozen |
| Score threshold | 0.05 |

Checkpoint directory:

```text
outputs/baselines/maskrcnn/maskrcnn_resnet50_fpn_cogar_full/
```

Result files:

```text
outputs/results/maskrcnn_resnet50_fpn_cogar_full.csv
outputs/tables/maskrcnn_resnet50_fpn_cogar_full_summary.json
outputs/tables/maskrcnn_resnet50_fpn_cogar_full_summary.csv
outputs/tables/maskrcnn_resnet50_fpn_cogar_full_per_class.csv
```

## Evaluation protocol

The model was evaluated using the project-style instance-mask protocol.

For each ground-truth object, the best class-matched predicted mask was selected. Mask IoU and boundary F1 were then computed.

This is not official COCO AP. It is an instance-level IoU comparison designed to align with the SAM, SAM2, and FastSAM evaluation protocol used elsewhere in this project.

## Summary results

| Metric | Value |
|---|---:|
| Test images | 75 |
| Test objects | 679 |
| Mean IoU | 0.7462 |
| Median IoU | 0.8309 |
| Mean boundary F1 | 0.7218 |
| IoU >= 0.90 | 0.2872 |
| IoU >= 0.75 | 0.6745 |
| IoU >= 0.50 | 0.8748 |
| IoU < 0.10 | 0.0604 |
| Total inference time | 13.4276 s |
| Image FPS | 5.5855 |
| Object FPS | 50.5675 |

## Per-class results

| Class | Objects | Mean IoU | Median IoU | IoU >= 0.50 | IoU < 0.10 |
|---|---:|---:|---:|---:|---:|
| box | 54 | 0.8124 | 0.9025 | 0.9074 | 0.0370 |
| cable | 42 | 0.5602 | 0.6104 | 0.6905 | 0.0952 |
| connector | 86 | 0.7575 | 0.8611 | 0.8721 | 0.0698 |
| glass_object | 54 | 0.6852 | 0.7897 | 0.8148 | 0.1296 |
| metal_part | 78 | 0.8288 | 0.8907 | 0.9359 | 0.0256 |
| plastic_object | 97 | 0.8499 | 0.9262 | 0.9381 | 0.0515 |
| robot_gripper | 155 | 0.7487 | 0.8066 | 0.9097 | 0.0258 |
| screw | 66 | 0.6396 | 0.7590 | 0.8333 | 0.1364 |
| tool | 47 | 0.6755 | 0.7216 | 0.7872 | 0.0426 |

## Interpretation

Mask R-CNN achieved a mean IoU of **0.7462** and a median IoU of **0.8309** on the full 75-image test split.

The strongest categories were:

- plastic_object: mean IoU 0.8499
- metal_part: mean IoU 0.8288
- box: mean IoU 0.8124

The most difficult categories were:

- cable: mean IoU 0.5602
- screw: mean IoU 0.6396
- tool: mean IoU 0.6755
- glass_object: mean IoU 0.6852

This pattern is consistent with the broader benchmark: thin, small, reflective, transparent, and irregular objects are harder than larger rigid objects.

## Comparison role

This result gives Task 5 an additional classical supervised instance-segmentation baseline.

YOLOv8n-seg remains the main supervised baseline because it reports standard detection and segmentation mAP metrics. Mask R-CNN is reported using the project-style class-matched mask IoU protocol, making it useful for qualitative comparison with the promptable segmentation models.

## Final status

Mask R-CNN is completed as a full-test-split supervised baseline for Assignment Task 5.

DeepLabV3+ remains documented as excluded because it is primarily a semantic segmentation model, while this project evaluates object-instance segmentation.

## References

- He et al., "Mask R-CNN", ICCV 2017. https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf
- TorchVision Mask R-CNN documentation. https://docs.pytorch.org/vision/main/models/mask_rcnn.html