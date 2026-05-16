# Assignment Task 4: Zero-Shot Prompt-Based Segmentation Benchmark

## Requirement

Evaluate zero-shot segmentation models on the simulated robotic-scene benchmark
using prompt-based and automatic-mask modes.

## Final status

Task 4 is complete for the full 4,471-object benchmark for:

- SAM ViT-B: box, point, and automatic mask generation.
- FastSAM-S: box, point, and automatic/everything mode.
- SAM2.1-Tiny: box, point, and automatic mask generation.

SAM ViT-H was evaluated in box, point, and automatic modes on CPU subsets only.
Full CUDA evaluation is hardware-limited on the available NVIDIA GTX 1050 4 GB
GPU due to CUDA out-of-memory during image encoding.

## Prompt-mode coverage

| Model | Box | Point | Auto / Everything | Scope |
|---|---|---|---|---|
| SAM ViT-B | Complete | Complete | Complete | Full 4,471 objects |
| FastSAM-S | Complete | Complete | Complete | Full 4,471 objects |
| SAM2.1-Tiny | Complete | Complete | Complete | Full 4,471 objects |
| SAM ViT-H | CPU subset | CPU subset | CPU subset | Hardware-limited reference |

## Visual evidence

![Mean IoU by model and prompt mode](/outputs/figures/final_report/metrics/mean_iou_by_model_prompt.png)

*Figure: Mean IoU across zero-shot prompt modes. SAM ViT-H entries are CPU subset results only.*

![Boundary F1 by model and prompt mode](/outputs/figures/final_report/metrics/boundary_f1_by_model_prompt.png)

*Figure: Boundary F1 comparison across the same prompt modes, highlighting boundary quality on small and thin objects.*

## Full-dataset results

### SAM ViT-B

| Prompt type | Objects | Mean IoU | Median IoU | Mean boundary F1 | IoU >= 0.75 | IoU < 0.10 |
|---|---:|---:|---:|---:|---:|---:|
| Box | 4,471 | 0.9057 | 0.9553 | 0.9356 | 0.9137 | 0.0031 |
| Point | 4,471 | 0.7985 | 0.9125 | 0.8131 | 0.7271 | 0.0204 |
| Auto | 4,471 | 0.8025 | 0.9422 | 0.8381 | 0.7486 | 0.0552 |

### FastSAM-S

| Prompt type | Objects | Mean IoU | Median IoU | Mean boundary F1 | IoU >= 0.75 | IoU < 0.10 | Mean FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Box | 4,471 | 0.698569 | 0.813478 | 0.891956 | 0.592709 | 0.082979 | 471.295280 |
| Point | 4,471 | 0.759372 | 0.888325 | 0.788963 | 0.710356 | 0.073809 | 214.069901 |
| Auto / Everything | 4,471 | 0.777331 | 0.891437 | 0.809290 | 0.720197 | 0.050772 | 206.475290 |

FastSAM point and auto/everything results use candidate-mask selection. FastSAM
first generates image-level candidate masks. Point evaluation selects candidate
masks containing the positive object prompt point, while auto/everything matches
the best candidate mask to each ground-truth object instance.

### SAM2.1-Tiny

| Prompt type | Objects | Mean IoU | Median IoU | Mean boundary F1 | IoU >= 0.75 | IoU < 0.10 | Mean FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Box | 4,471 | 0.912746 | 0.955280 | 0.930659 | 0.926191 | 0.000895 | 16.809629 |
| Point | 4,471 | 0.865783 | 0.934924 | 0.873056 | 0.827555 | 0.004921 | 16.679109 |
| Auto | 4,471 | 0.640259 | 0.870136 | 0.678148 | 0.586670 | 0.224782 | 2.300666 |

SAM2.1-Tiny was installed in a separate environment under
`/mnt/Info/COGAR_Large/SAM2/` to avoid modifying the main benchmark
environment.

## SAM ViT-H hardware-limited results

SAM ViT-H was re-tested for all three required prompt modes. Because full CUDA
evaluation failed on the available NVIDIA GTX 1050 4 GB GPU, SAM ViT-H is
reported as a CPU subset feasibility/reference result rather than a full
4,471-object benchmark.

| Prompt mode | Subset | Device | Mean IoU | Median IoU | Mean boundary F1 | Mean FPS |
|---|---:|---|---:|---:|---:|---:|
| Box | 25 objects | CPU | 0.9449 | 0.9717 | 0.9637 | 0.1820 |
| Point | 25 objects | CPU | 0.7721 | 0.9547 | 0.7958 | 0.1762 |
| Auto | 42 objects / 5 images | CPU | 0.7302 | 0.9563 | 0.7640 | 0.2118 |

The ViT-H GPU failure occurred during image encoding in
`predictor.set_image(image)`, before object-level prompt prediction. Reducing the
number of object prompts therefore did not resolve the memory limit.

## Interpretation

SAM2.1-Tiny box prompting achieved the highest mean IoU among the completed
full-dataset promptable box-prompt evaluations. SAM ViT-B box remains a strong
speed/accuracy balance in the current implementation. FastSAM-S is the fastest
SAM-style model, but its box-prompt masks are less accurate.

Automatic mask generation is harder than prompted segmentation because the model
must discover object masks without instance prompts. This is especially visible
for SAM2.1-Tiny auto mode on screws, small parts, and robot-gripper-heavy
scenes.

## Supporting result pages

- [Final SAM ViT-B results](final_sam_vit_b_results.md)
- [Final SAM ViT-H results](final_sam_vit_h_results.md)
- [Final SAM2.1-Tiny results](final_sam2_results.md)
- [Final FastSAM-S results](final_fastsam_results.md)
- [Final cross-model results](final_cross_model_results.md)
