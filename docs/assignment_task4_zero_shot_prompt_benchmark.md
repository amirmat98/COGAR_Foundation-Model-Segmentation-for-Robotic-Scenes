# Assignment Task 4: Zero-Shot Prompt-Based Segmentation Benchmark

## FastSAM-S prompt-mode completion update

FastSAM-S was extended from box-only evaluation to three prompt modes:

| Prompt type | Objects | Mean IoU | Median IoU | Mean boundary F1 | IoU >= 0.75 | IoU < 0.10 | Mean FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Box | 4,471 | 0.698569 | 0.813478 | 0.891956 | 0.592709 | 0.082979 | 471.295280 |
| Point | 4,471 | 0.759372 | 0.888325 | 0.788963 | 0.710356 | 0.073809 | 214.069901 |
| Auto / Everything | 4,471 | 0.777331 | 0.891437 | 0.809290 | 0.720197 | 0.050772 | 206.475290 |

The point and auto/everything results use a candidate-mask selection protocol. FastSAM first generates image-level candidate masks. Point evaluation selects masks containing the object prompt point, while auto/everything evaluation matches the best candidate mask to each ground-truth object instance.

This completes FastSAM-S prompt-mode coverage for the assignment-level benchmark.

## SAM2.1-Tiny completion update

SAM2.1-Tiny was successfully installed in a separate environment under `/mnt/Info/COGAR_Large/SAM2/` and evaluated on the full 4,471-object simulated dataset.

Completed SAM2 prompt modes:

| Model | Prompt type | Objects | Mean IoU | Median IoU | Mean boundary F1 | IoU >= 0.75 | IoU < 0.10 | Mean FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SAM2.1-Tiny | Box | 4,471 | 0.912746 | 0.955280 | 0.930659 | 0.926191 | 0.000895 | 16.809629 |
| SAM2.1-Tiny | Point | 4,471 | 0.865783 | 0.934924 | 0.873056 | 0.827555 | 0.004921 | 16.679109 |

This completes SAM2 box-prompt and point-prompt evaluation for the assignment.

SAM2 automatic mask generation was not included in the completed run and remains a limitation/future extension.

Updated Task 4 status:

- SAM ViT-B: box, point, and automatic mask generation completed.
- FastSAM-S: box, point, and automatic/everything prompt evaluation completed.
- SAM2.1-Tiny: box and point prompt evaluation completed.
- SAM ViT-H: small CPU subset completed with hardware caveat.
- SAM2 auto and full SAM ViT-H remain limitations due to hardware/runtime constraints.

## Final Task 4 completion update

Task 4 is now completed for SAM ViT-B, FastSAM-S, and SAM2.1-Tiny prompt-mode coverage.

Completed prompt modes:

| Model | Box | Point | Auto / Everything |
|---|---:|---:|---:|
| SAM ViT-B | Completed | Completed | Completed |
| FastSAM-S | Completed | Completed | Completed |
| SAM2.1-Tiny | Completed | Completed | Completed |
| SAM ViT-H | CPU subset only | Not completed | Not completed |

### SAM2.1-Tiny final prompt-mode results

| Prompt type | Objects | Mean IoU | Median IoU | Mean boundary F1 | IoU >= 0.75 | IoU < 0.10 | Mean FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Box | 4,471 | 0.912746 | 0.955280 | 0.930659 | 0.926191 | 0.000895 | 16.809629 |
| Point | 4,471 | 0.865783 | 0.934924 | 0.873056 | 0.827555 | 0.004921 | 16.679109 |
| Auto | 4,471 | 0.640259 | 0.870136 | 0.678148 | 0.586670 | 0.224782 | 2.300666 |

### Final interpretation

SAM2.1-Tiny is now fully evaluated with box, point, and automatic mask generation on the full simulated dataset.

SAM2.1-Tiny box prompting achieved the highest mean IoU among the completed full-dataset box-prompt benchmarks. SAM2.1-Tiny point prompting was also strong, while automatic mask generation was substantially weaker because it is prompt-free and must discover object masks without instance prompts.

The only remaining Task 4 limitation is SAM ViT-H. Full SAM ViT-H evaluation was not feasible on the GTX 1050 4 GB GPU, so only a small CPU subset was completed and documented with a caveat.

## SAM ViT-H completion update

SAM ViT-H was re-tested for all three required prompt modes: box prompts, point prompts, and automatic mask generation.

Because full CUDA evaluation failed on the available NVIDIA GTX 1050 4 GB GPU, SAM ViT-H is reported as a CPU subset feasibility/reference result rather than a full-dataset benchmark. The GPU failure occurred during image encoding in `predictor.set_image(image)`, before object-level prompt prediction. Therefore, reducing the number of object prompts did not solve the memory limit.

### SAM ViT-H CPU subset results

| Prompt mode | Subset | Device | Mean IoU | Median IoU | Mean boundary F1 | Mean FPS |
|---|---:|---|---:|---:|---:|---:|
| box | 25 objects | CPU | 0.9449 | 0.9717 | 0.9637 | 0.1820 |
| point | 25 objects | CPU | 0.7721 | 0.9547 | 0.7958 | 0.1762 |
| auto | 42 objects / 5 images | CPU | 0.7302 | 0.9563 | 0.7640 | 0.2118 |

### Final Task 4 status

Task 4 is completed for SAM ViT-B, SAM2.1-Tiny, and FastSAM-S on the full 4,471-object dataset across box prompts, point prompts, and automatic mask generation.

SAM ViT-H was evaluated in all three prompt modes on CPU subsets only due to confirmed GPU memory limitations. It is documented as a hardware-limited reference result and excluded from full cross-model ranking.
