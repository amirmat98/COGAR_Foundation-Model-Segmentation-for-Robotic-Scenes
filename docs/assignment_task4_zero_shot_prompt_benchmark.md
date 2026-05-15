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
