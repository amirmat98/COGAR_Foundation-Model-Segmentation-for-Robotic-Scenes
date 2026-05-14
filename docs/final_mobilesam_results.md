# Final MobileSAM Results on COGAR-SimRobotics-500

## Setup

MobileSAM was evaluated as a lightweight SAM-style segmentation model for edge/real-time robotic perception. It uses the MobileSAM checkpoint `mobile_sam.pt` and box prompts from the final benchmark index.

- Final dataset: COGAR-SimRobotics-500
- Images: 500
- Object instances evaluated: 4,471
- Prompt type: box
- Checkpoint: `checkpoints/mobile_sam.pt`

## SAM ViT-B vs MobileSAM box-prompt comparison

| Run | Objects | Mean IoU | Median IoU | Boundary F1 | IoU >= 0.90 | IoU >= 0.75 | IoU >= 0.50 | IoU < 0.10 | Mean FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM ViT-B box | 4,471 | 0.9057 | 0.9553 | 0.9356 | 0.7513 | 0.9137 | 0.9703 | 0.0031 | 61.96 |
| MobileSAM box | 4,471 | 0.8656 | 0.9363 | 0.9797 | 0.6285 | 0.8430 | 0.9457 | 0.0045 | 69.52 |

## Interpretation

MobileSAM provides a good lightweight trade-off for robotic perception. Compared with SAM ViT-B box prompting, MobileSAM has lower mean IoU and fewer objects above IoU 0.90, but it keeps high median IoU and a very low catastrophic failure rate.

The result suggests that MobileSAM is suitable when edge deployment or faster inference is more important than maximum segmentation accuracy. SAM ViT-B remains the stronger model when accuracy is the priority.

The reported FPS is an object-row-level benchmark from the evaluation script. Because image embeddings are reused for multiple objects from the same image, this FPS should be treated as a relative speed indicator rather than a strict per-frame robot runtime measurement.
