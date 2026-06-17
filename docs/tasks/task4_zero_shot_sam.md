# Task 4: Zero-Shot SAM-Family Segmentation

Task 4 runs promptable segmentation models without dataset-specific training.
The goal is to produce prediction files for later evaluation, not to fine-tune
the models.

Status: complete.

## Figures

![Zero-shot mIoU heatmap](../../outputs/final_benchmark_assets/plots/zero_shot_miou_heatmap.png)

![Best zero-shot model by dataset and prompt](../../outputs/final_benchmark_assets/plots/zero_shot_dataset_prompt_winners.png)

## Models

The benchmark includes:

- SAM ViT-H
- SAM ViT-B
- SAM2
- FastSAM

## Prompt Modes

Each model is evaluated in three modes where supported:

- Point prompt: one positive point generated from the ground-truth instance
  mask. The point is the foreground mask pixel closest to the mask centroid.
- Box prompt: one box generated from the ground-truth COCO bounding box.
- Automatic mask generation: no ground-truth prompt is given; the model
  proposes masks for the whole image.

Point and box modes test how well each model segments a known target when a
robotic perception system provides a simple spatial cue. Automatic mode tests
whether the model can discover object masks without prompts.

## Active Datasets

The Task 4 runner uses datasets that have COCO-style instance annotations:

- Isaac official Unitree G1
- BlenderProc COGAR-SimRobotics-1000
- OCID

OCID originally uses integer instance-label PNG files. It is converted to
COCO-style instance annotations with `scripts/datasets/convert_ocid_to_coco.py`
before using the same prompt runner.

## Outputs

Prompt manifests are written to:

```text
outputs/task4_zero_shot_sam/prompts
```

Model predictions were written to:

```text
results/task4_zero_shot_sam
```

Large prediction files, model checkpoints, and generated outputs stay outside
Git.

## Environment

Use `requirements.txt` for lightweight local setup and prompt-manifest
validation. Use `requirements-task4-gpu.txt` on the GPU machine after
installing the matching PyTorch and TorchVision CUDA wheels.

## Closure Criteria

Task 4 is complete when prediction files exist for each enabled model, dataset,
and prompt mode, and each run records the model checkpoint, dataset version,
device, prompt mode, and runtime metadata.

The completed benchmark produced predictions for 4 zero-shot model variants,
3 prompt modes, and 3 datasets. Compact prompt manifest summaries are stored in
`outputs/task4_zero_shot_sam/prompts/summary.json`.
