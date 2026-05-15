# SAM ViT-H Hardware Limitation Note

SAM ViT-H was attempted on the available NVIDIA GTX 1050 4 GB GPU.

## GPU attempts

The following configurations were tested:

- CUDA FP32 box prompt, 5 objects
- CUDA FP16 box prompt, 5 objects
- CUDA AMP/mixed-precision box prompt, 5 objects

The FP32 and AMP runs failed during `predictor.set_image(image)`, specifically inside the SAM ViT-H image encoder. This means the memory bottleneck occurs before object prompt prediction, so reducing the number of object prompts does not solve the problem.

The FP16 run failed because forcing the full SAM model to half precision caused a dtype mismatch in the prompt encoder.

## CPU subset results

| Prompt mode | Subset | Device | Mean IoU | Median IoU | Mean boundary F1 | Mean FPS |
|---|---:|---|---:|---:|---:|---:|
| box | 25 objects | CPU | 0.9449 | 0.9717 | 0.9637 | 0.1820 |
| point | 25 objects | CPU | 0.7721 | 0.9547 | 0.7958 | 0.1762 |
| auto | 42 objects / 5 images | CPU | 0.7302 | 0.9563 | 0.7640 | 0.2118 |

## Conclusion

Full SAM ViT-H evaluation was not feasible on the GTX 1050 4 GB GPU. Therefore, SAM ViT-H is reported only as a CPU subset feasibility/reference result. It is excluded from full cross-model ranking, while SAM ViT-B, SAM2.1-Tiny, and FastSAM-S are evaluated on the full 4,471-object dataset across box, point, and automatic mask generation modes.
