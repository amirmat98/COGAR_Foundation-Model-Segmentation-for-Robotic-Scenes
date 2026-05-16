# SAM ViT-H Hardware-Limited Evaluation

## Role in the benchmark

SAM ViT-H is included as the large SAM reference model. It was tested for the
same prompt modes used elsewhere in the benchmark:

- box prompts
- point prompts
- automatic mask generation

Because the available GPU is an NVIDIA GTX 1050 with 4 GB VRAM, SAM ViT-H could
not be evaluated on the full 4,471-object dataset with CUDA. The final ViT-H
results are therefore CPU subset results only.

## Visual evidence

![SAM ViT-H CPU subset in mean IoU chart](/outputs/figures/final_report/metrics/mean_iou_by_model_prompt.png)

*Figure: SAM ViT-H box, point, and auto CPU subset results are labelled separately from full-dataset GPU runs.*

![SAM ViT-H CPU FPS in speed chart](/outputs/figures/final_report/speed/fps_comparison.png)

*Figure: SAM ViT-H CPU subset speed appears at the low end of the log-scale FPS comparison, supporting the hardware-limited conclusion.*

## Checkpoint verification

The official SAM ViT-H checkpoint was downloaded and verified successfully.

| Item | Value |
|---|---|
| Checkpoint | `checkpoints/sam_vit_h_4b8939.pth` |
| Size | 2,564,550,879 bytes |
| PyTorch load test | successful |
| Number of tensors | 594 |

The checkpoint is not committed to the repository.

## CUDA limitation

Full SAM ViT-H GPU evaluation failed during image encoding in
`predictor.set_image(image)`. This happens before object-level prompt prediction,
so reducing the number of prompts per image does not remove the main memory
pressure.

Observed hardware/runtime issues:

- CUDA FP32 evaluation raised out-of-memory errors on the GTX 1050 4 GB GPU.
- CUDA AMP/mixed precision still exceeded the practical memory limit.
- FP16 attempts exposed dtype mismatch issues in the local environment.

For this reason, SAM ViT-H is excluded from the full cross-model ranking and
reported only as a hardware-limited reference.

## CPU subset results

| Prompt mode | Subset | Device | Mean IoU | Median IoU | Mean boundary F1 | Mean FPS |
|---|---:|---|---:|---:|---:|---:|
| Box | 25 objects | CPU | 0.9449 | 0.9717 | 0.9637 | 0.1820 |
| Point | 25 objects | CPU | 0.7721 | 0.9547 | 0.7958 | 0.1762 |
| Auto | 42 objects / 5 images | CPU | 0.7302 | 0.9563 | 0.7640 | 0.2118 |

## Interpretation

The ViT-H checkpoint and implementation are valid, and the CPU subset confirms
that all three prompt modes can run. However, CPU throughput is far too slow for
the complete 4,471-object benchmark, and CUDA execution is not feasible on the
available 4 GB GPU.

For the final assignment, SAM ViT-B is the complete SAM-family full-dataset
reference, while SAM ViT-H is documented as a larger hardware-limited model.
