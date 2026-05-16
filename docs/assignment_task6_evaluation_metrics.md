# Assignment Task 6: Evaluation Metrics

Task 6 is completed.

The benchmark evaluates segmentation quality using mean IoU, boundary F1, mask AP, and per-category IoU.

## Metric coverage

| Required metric | Status | Evidence |
|---|---|---|
| Mean IoU / mIoU | Completed | Reported for SAM ViT-B, SAM2.1-Tiny, FastSAM-S, SAM ViT-H subset, and Mask R-CNN |
| Boundary F1 | Completed | Reported for zero-shot promptable models and Mask R-CNN |
| Mask AP | Completed | Reported for YOLOv8n-seg as mask precision, recall, mAP50, and mAP50-95 |
| Per-category IoU | Completed | Reported for SAM2.1-Tiny, Mask R-CNN, and cross-model category comparisons |

## Notes

Mask AP is reported mainly for YOLOv8n-seg because it is a supervised confidence-scored instance-segmentation detector.

SAM, SAM2, and FastSAM are evaluated primarily with object-level IoU and boundary F1 because they are prompt-conditioned or automatic-mask-generation models rather than class-labelled confidence-ranked detectors in this benchmark.

DeepLabV3+ is not included in the instance-level metric table because it is a semantic segmentation model, while this project evaluates object instances.

## Conclusion

Task 6 is complete. The project includes standard segmentation metrics for overlap quality, boundary quality, category-level performance, and supervised instance-segmentation AP.
