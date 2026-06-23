# State of the Art

The benchmark is positioned between general-purpose segmentation research and
robotic deployment requirements.

| Stream | Methods | Role |
|---|---|---|
| Promptable foundation segmentation | SAM, SAM ViT-H, SAM ViT-B | Zero-shot segmentation from prompts. |
| Temporal foundation segmentation | SAM2 | Segmentation direction relevant to video/continuous perception. |
| Lightweight SAM variants | FastSAM, MobileSAM, EfficientSAM | Lower-cost segmentation for deployment trade-offs. |
| Supervised baselines | Mask R-CNN, DeepLabV3+, YOLOv8-seg | Task-domain adaptation and real-time comparison. |
| Simulation and robotic data | Isaac Sim, BlenderProc, OCID | Controlled annotations and robotic challenge coverage. |

Main gap:

> Strong segmentation on generic images does not prove reliable robotic
> perception under transparent, reflective, occluded, small-object, robot-body,
> dynamic, and real-time constraints.

More detail:

- [../../report/02_state_of_the_art.md](../../report/02_state_of_the_art.md)
- [../../REPORT.md#2-state-of-the-art](../../REPORT.md#2-state-of-the-art)
- [../../report/references.md](../../report/references.md)
