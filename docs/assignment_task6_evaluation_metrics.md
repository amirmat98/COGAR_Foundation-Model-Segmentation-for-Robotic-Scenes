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

## Visual evidence

![Mean IoU by model and prompt](/outputs/figures/final_report/metrics/mean_iou_by_model_prompt.png)

*Figure: Mean IoU is the primary overlap-quality metric for promptable and automatic-mask runs.*

![Boundary F1 by model and prompt](/outputs/figures/final_report/metrics/boundary_f1_by_model_prompt.png)

*Figure: Boundary F1 summarizes contour quality, which is important for cables, screws, connectors, and thin robotic parts.*

![Per-category IoU comparison](/outputs/figures/final_report/metrics/per_category_iou.png)

*Figure: Per-category IoU comparison using existing SAM2.1-Tiny and Mask R-CNN category tables.*

![Supervised baseline AP and IoU summary](/outputs/figures/final_report/metrics/supervised_baselines_summary.png)

*Figure: Supervised baseline chart showing YOLOv8n-seg mask AP metrics and Mask R-CNN IoU/BF1 metrics in separate panels.*

## Notes

Mask AP is reported mainly for YOLOv8n-seg because it is a supervised confidence-scored instance-segmentation detector.

SAM, SAM2, and FastSAM are evaluated primarily with object-level IoU and boundary F1 because they are prompt-conditioned or automatic-mask-generation models rather than class-labelled confidence-ranked detectors in this benchmark.

DeepLabV3+ is not included in the instance-level metric table because it is a semantic segmentation model, while this project evaluates object instances.

## Conclusion

Task 6 is complete. The project includes standard segmentation metrics for overlap quality, boundary quality, category-level performance, and supervised instance-segmentation AP.
