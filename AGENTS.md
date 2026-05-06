# AGENTS.md

## Project Context

This repository supports the COGAR I2 direction: a zero-shot segmentation
benchmark for robotic scenes using OCID and COGAR-SimRobotics-500, with SAM now
and SAM2, FastSAM, MobileSAM, and EfficientSAM support planned.

## Layout

- `src/cogar_seg/`: reusable package code.
- `scripts/`: thin CLI wrappers only.
- `configs/`: reproducible configs and local path examples.
- `docs/`: benchmark, dataset, and setup notes.
- `tests/`: lightweight tests that do not require datasets or checkpoints.
- `data/`, `outputs/`, `checkpoints/`: local/generated assets ignored by Git.

## Generated-Data Policy

Do not delete local datasets, checkpoints, generated outputs, or experiment
artifacts. Do not commit raw generated images, COCO outputs, checkpoints, masks,
plots, or result CSVs. If generated/local files are tracked, remove them only
with `git rm --cached`, then keep placeholders such as `data/README.md` and
`outputs/README.md` tracked.

## Scripts vs Package Code

Keep scripts as small command-line entry points that parse arguments and call
functions from `src/cogar_seg/`. Put reusable logic under clear package modules
for datasets, generation, indexing, evaluation, metrics, visualization, prompts,
and model wrappers.

## Common Commands

Generate COGAR-Sim with BlenderProc:

```bash
source ~/blenderproc_test/.venv/bin/activate
blenderproc run scripts/blenderproc/generate_cogar_sim_500.py \
  --config configs/blenderproc_dataset.yaml \
  --num-images 500
```

Normalize generated data:

```bash
PYTHONPATH=src python scripts/dataset/normalize_cogar_sim_500.py
```

Create COGAR-Sim object index:

```bash
PYTHONPATH=src python scripts/dataset/create_object_index.py \
  --dataset cogar_sim_500 \
  --coco data/cogar_sim_500/annotations/instances_all.json \
  --metadata data/cogar_sim_500/metadata/frame_index.csv \
  --rgb-dir data/cogar_sim_500/rgb \
  --output outputs/indexes/cogar_sim_500_objects.csv
```

Run validation:

```bash
python -m py_compile $(find src scripts -name "*.py")
PYTHONPATH=src pytest -q
```

## Coding Conventions

Prefer small functions, dataclasses for structured run outputs, explicit path
validation, and clear errors. Keep imports of heavy dependencies inside the
functions that need them when possible. Do not add console scripts until the
package entry points are stable.

## Future Codex Workflow

Start by checking `git status --short`, current branch, and relevant tests.
Preserve working OCID and BlenderProc flows. Archive uncertain legacy code under
`archive/YYYYMMDD_cleanup/` rather than deleting it. Before final reporting, run
syntax checks, tests, `git status --short`, and `git ls-files` checks for ignored
local assets.
