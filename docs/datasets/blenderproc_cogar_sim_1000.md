# Dataset: BlenderProc COGAR-SimRobotics-1000

## Summary

- Type: synthetic tabletop robotic-scene dataset
- Simulator: BlenderProc
- Images: 1000 target
- Resolution: 640x480
- Classes: table, robot gripper, metal part, glass object, plastic object,
  screw, connector, cable, tool, box

## Output Paths

Generation target on this machine:

```text
/mnt/Info/COGAR_DATASETs/BlenderProc_cogar_sim_1000
```

Default repo path for another user:

```text
Datasets/BlenderProc_cogar_sim_1000
```

Users may store it elsewhere and update `configs/datasets.yaml`.

## Generate

```bash
blenderproc run scripts/blenderproc/generate_cogar_sim.py \
  --config configs/blenderproc_dataset.yaml
```

Normalize:

```bash
.venv/bin/python scripts/datasets/normalize_blenderproc_cogar_sim.py
```

AWS runbook:

```text
docs/datasets/blenderproc_aws_runbook.md
```

## Smoke Test

Validation date: 2026-06-13

- Raw dataset: `smoke_5_clean`
- Images: 5
- COCO annotations: 53
- COCO categories: 10
- Metadata rows: 5
- Result: PASS

## Final Dataset Target

- RGB images: 1000
- COCO categories: 10
- Split: 700 train, 150 validation, 150 test
- Challenge balance: 200 images per challenge family
- Generation target: AWS
- Result: pending

## Release URL

`TODO_PUBLIC_URL`
