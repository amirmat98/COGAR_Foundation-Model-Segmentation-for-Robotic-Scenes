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
