# Task 4: Zero-Shot SAM-Family Segmentation

Task 4 runs promptable segmentation models without dataset-specific training.
The goal is to produce prediction files for later evaluation, not to fine-tune
the models.

Status: started.

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

The first Task 4 runner uses datasets that already have COCO instance
annotations:

- Isaac official Unitree G1
- BlenderProc COGAR-SimRobotics-1000

OCID is kept for the benchmark, but it needs a separate conversion from its
label images to COCO-style instance annotations before using the same prompt
runner.

## Outputs

Prompt manifests are written to:

```text
outputs/task4_zero_shot_sam/prompts
```

Model predictions will be written to:

```text
results/task4_zero_shot_sam
```

Large prediction files, model checkpoints, and generated outputs stay outside
Git.

## Closure Criteria

Task 4 is complete when prediction files exist for each enabled model, dataset,
and prompt mode, and each run records the model checkpoint, dataset version,
device, prompt mode, and runtime metadata.
