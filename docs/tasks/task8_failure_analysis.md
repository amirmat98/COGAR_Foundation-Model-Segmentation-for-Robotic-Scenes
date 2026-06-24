# Task 8 - Failure Mode Analysis

Task 8 identifies where and why the segmentation models fail in robotic scenes.

## Figures

![Robotic challenge group performance](../../outputs/final_benchmark_assets/plots/challenge_group_weighted_iou.png)

![Representative failure overlay](../../outputs/task8_failure_analysis/test/figures/04_02_blenderproc_cogar_sim_sam_vit_h_automatic_glass_object_iou_0.000.png)

## Inputs

The analysis uses existing benchmark artifacts:

- Task 4 prediction JSONL files in `results/task4_zero_shot_sam/`
- Task 6 held-out test summaries in `outputs/task6_evaluation/zero_shot/test/`
  and `outputs/task6_evaluation/baselines/test/`
- Task 7 speed summaries in `outputs/task7_inference_speed/`
- COCO annotations and source images from the configured datasets

## Method

The analysis is metric-driven first, then qualitative:

1. Expand Task 6 per-category IoU and boundary F1 into a single failure table.
2. Group categories into robotic challenge groups:
   - small parts and thin structures
   - transparent or reflective surfaces
   - robot body, moving objects, and occlusion
   - scene support/background objects
3. Compare point, box, and automatic prompt modes for each zero-shot model.
4. Join Task 6 quality metrics with Task 7 FPS to separate segmentation quality from real-time feasibility.
5. Mine representative low-IoU examples from selected Task 4 prediction files,
   restricted to the same held-out test image IDs.
6. Save overlay images where green is ground truth, red is prediction, and yellow is overlap.

Challenge groups are derived from COCO category names and dataset design
metadata. They are valid for reporting object-family robustness, but they are
not a substitute for per-instance physical labels such as exact reflectance,
transparency, occlusion ratio, or motion blur. Claims in the final report should
therefore describe these tables as challenge-group evidence plus qualitative
examples, not as a fully instrumented material/physics ablation.

## Command

```bash
cd ~/Desktop/COGAR/COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes
source .venv/bin/activate

.venv/bin/python scripts/analysis/analyze_task8_failure_modes.py
```

The script prints progress while mining visual examples.

## Outputs

Compact outputs are written to:

```text
outputs/task8_failure_analysis/test/
```

The complete raw `results/` folder used as analysis input could not be included
in Git because it is too large. The derived failure tables and selected visual
examples are committed under `outputs/`.

Important files:

```text
outputs/task8_failure_analysis/test/task8_failure_analysis.md
outputs/task8_failure_analysis/test/category_failures.csv
outputs/task8_failure_analysis/test/challenge_group_summary.csv
outputs/task8_failure_analysis/test/prompt_mode_comparison.csv
outputs/task8_failure_analysis/test/speed_quality_tradeoff.csv
outputs/task8_failure_analysis/test/representative_failures.csv
outputs/task8_failure_analysis/test/figures/
```

The script fails if Task 6 inputs do not contain `split=test`,
`evaluation_images`, and `split_sha256` metadata. The generated CSV files carry
the same fields so downstream plots can prove they use the identical held-out
test image IDs.

The generated report is intended to be copied into the final project report after review.
