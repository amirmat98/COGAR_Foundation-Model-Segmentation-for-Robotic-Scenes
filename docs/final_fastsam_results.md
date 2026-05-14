# Final FastSAM-S Results on COGAR-SimRobotics-500

## Setup

FastSAM-S was evaluated as a fast CNN-based Segment Anything alternative for robotic-scene segmentation.

- Final dataset: COGAR-SimRobotics-500
- Images: 500
- Object instances evaluated: 4,471
- Prompt type: box-style selection
- Checkpoint: `checkpoints/FastSAM-s.pt`

## Result

| Model | Objects | Mean IoU | Median IoU | Boundary F1 | Mean FPS |
|---|---:|---:|---:|---:|---:|
| FastSAM-S box | 4,471 | 0.6986 | 0.8135 | 0.8920 | 471.30 |

## Interpretation

FastSAM-S is much faster than the SAM-style transformer models in this evaluation script, but it gives lower segmentation accuracy. Compared with SAM ViT-B and MobileSAM, FastSAM-S has a substantially lower mean IoU and median IoU.

This makes FastSAM-S useful as a real-time or resource-constrained baseline, but less suitable when accurate object boundaries are required for robotic manipulation or fine-grained perception. In this benchmark, FastSAM-S is best interpreted as the high-speed/low-accuracy point of the trade-off curve.

The reported FPS is an object-row-level benchmark from the evaluation script. Because image-level predictions are reused for multiple target objects from the same image, this FPS should be treated as a relative speed indicator rather than a strict per-frame robot runtime measurement.
