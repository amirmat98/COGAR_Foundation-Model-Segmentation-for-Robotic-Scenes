# FastSAM-S Zero-Shot Results

## Role in benchmark

FastSAM-S is included as a speed-first Segment Anything-style zero-shot segmentation baseline.

It was evaluated on the final simulated robotic-scene dataset:

- `data/cogar_sim_500_final/annotations/sim_robotic_scenes_index_final_filtered.csv`

## Visual evidence

![FastSAM-S prompt modes in mean IoU chart](/outputs/figures/final_report/metrics/mean_iou_by_model_prompt.png)

*Figure: FastSAM-S box, point, and auto/everything results shown in the zero-shot prompt-mode comparison.*

![FastSAM-S speed in FPS chart](/outputs/figures/final_report/speed/fps_comparison.png)

*Figure: FastSAM-S has the highest measured zero-shot throughput in the speed comparison.*

## Evaluation protocol

FastSAM-S was evaluated with three prompt modes:

- box prompt
- point prompt
- automatic/everything mode

The box-prompt result uses the previously completed object-level box-prompt protocol.

The point and automatic/everything results use a candidate-mask selection protocol:

- FastSAM first generates candidate masks for each image.
- For point prompting, masks containing the positive object prompt point are selected.
- For automatic/everything evaluation, the best candidate mask is matched to each ground-truth instance for scoring.
- IoU and boundary F1 are then computed against the ground-truth instance masks.

This protocol makes the FastSAM point and auto results comparable as object-level benchmark scores, but they should be interpreted as candidate-mask selection results rather than identical decoder behavior to SAM ViT-B.

## Overall results

| Prompt type | Objects | Mean IoU | Median IoU | Mean boundary F1 | IoU >= 0.90 | IoU >= 0.75 | IoU >= 0.50 | IoU < 0.10 | Mean FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Box | 4471 | 0.698569 | 0.813478 | 0.891956 | 0.284053 | 0.592709 | 0.808544 | 0.082979 | 471.295280 |
| Point | 4471 | 0.759372 | 0.888325 | 0.788963 | 0.473049 | 0.710356 | 0.840081 | 0.073809 | 214.069901 |
| Auto / Everything | 4471 | 0.777331 | 0.891437 | 0.809290 | 0.479087 | 0.720197 | 0.863565 | 0.050772 | 206.475290 |

## Interpretation

FastSAM-S provides the highest speed among the evaluated Segment Anything-style models, but with lower accuracy than SAM ViT-B and MobileSAM.

Main observations:

- FastSAM-S box prompting achieved very high throughput but lower mean IoU than SAM ViT-B box and MobileSAM box.
- FastSAM-S point and auto/everything evaluations improved mean IoU under the candidate-mask matching protocol.
- The point and auto/everything scores should be interpreted carefully because they are based on selecting from generated candidate masks.
- FastSAM-S remains the strongest speed-first baseline in the project.

## Task 4 relevance

FastSAM-S now has full prompt-mode coverage for the assignment-level comparison:

- box prompt
- point prompt
- automatic/everything mode

Generated result files are intentionally not committed:

- `outputs/tables/fastsam/point_final/`
- `outputs/tables/fastsam/auto_final/`
