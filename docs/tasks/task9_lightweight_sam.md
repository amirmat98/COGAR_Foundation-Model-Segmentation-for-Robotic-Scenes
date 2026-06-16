# Task 9: Lightweight SAM Edge-Deployment Trade-Off

## Goal

Task 9 tests whether lightweight SAM variants can give a useful accuracy and
speed trade-off for robotic perception on the same benchmark datasets used in
Tasks 4-8.

The evaluated lightweight models are:

| Model | Source | Prompt modes |
| --- | --- | --- |
| MobileSAM ViT-T | <https://github.com/ChaoningZhang/MobileSAM> | point, box, automatic |
| EfficientSAM-Ti | <https://github.com/yformer/EfficientSAM> | point, box, grid automatic |
| EfficientSAM-S | <https://github.com/yformer/EfficientSAM> | point, box, grid automatic |

MobileSAM keeps the SAM-compatible predictor and automatic-mask-generator
interface. EfficientSAM exposes direct point/box tensor inference; therefore
its automatic setting is implemented as a regular grid of positive point
prompts and is reported as grid automatic proposal generation.

## Outputs

Task 9 keeps all generated files separate from completed Task 4 outputs:

| Output | Path |
| --- | --- |
| Predictions | `results/task9_lightweight_sam/` |
| Evaluation metrics | `outputs/task9_lightweight_sam/evaluation/` |
| Speed metrics | `outputs/task9_lightweight_sam/inference_speed/` |
| Trade-off report | `outputs/task9_lightweight_sam/summary/task9_lightweight_sam_report.md` |

## Setup

Use the same Python virtual environment and dataset paths as Tasks 4-8.

```bash
source .venv/bin/activate
python -m pip install -r requirements-task9-gpu.txt
```

Download the lightweight checkpoints:

```bash
mkdir -p checkpoints/mobile_sam checkpoints/efficient_sam

wget -nc -O checkpoints/mobile_sam/mobile_sam.pt \
  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

wget -nc -O checkpoints/efficient_sam/efficient_sam_vitt.pt \
  https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vitt.pt

wget -nc -O checkpoints/efficient_sam/efficient_sam_vits.pt.zip \
  https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vits.pt.zip

unzip -n checkpoints/efficient_sam/efficient_sam_vits.pt.zip \
  -d checkpoints/efficient_sam

if [ -f checkpoints/efficient_sam/weights/efficient_sam_vits.pt ]; then
  mv checkpoints/efficient_sam/weights/efficient_sam_vits.pt \
    checkpoints/efficient_sam/efficient_sam_vits.pt
fi
```

## Smoke Test

Run a small inference smoke test first:

```bash
python scripts/benchmarks/run_zero_shot_sam.py \
  --config configs/task9_lightweight_sam.yaml \
  --dataset blenderproc_cogar_sim \
  --model mobile_sam_vit_t \
  --prompt-mode point \
  --max-instances 1
```

Then check the dry-run plan for the full dataset batch:

```bash
python scripts/benchmarks/run_task4_dataset_batch.py \
  --config configs/task9_lightweight_sam.yaml \
  --dataset blenderproc_cogar_sim \
  --dry-run
```

## Full Inference

Run the three datasets one after another:

```bash
mkdir -p logs/task9

nohup bash -lc '
set -euo pipefail
export PYTHONUNBUFFERED=1

for dataset in blenderproc_cogar_sim ocid isaac_official_unitree_g1; do
  echo "===== TASK 9 DATASET START: ${dataset} ====="
  python scripts/benchmarks/run_task4_dataset_batch.py \
    --config configs/task9_lightweight_sam.yaml \
    --dataset "${dataset}" \
    --log-every 250
  echo "===== TASK 9 DATASET DONE: ${dataset} ====="
done
' > logs/task9/lightweight_sam_full.log 2>&1 &

echo $! > logs/task9/lightweight_sam_full.pid
tail -f logs/task9/lightweight_sam_full.log
```

## Evaluation

```bash
nohup bash -lc '
set -euo pipefail
export PYTHONUNBUFFERED=1

python scripts/evaluation/evaluate_task6_zero_shot.py \
  --config configs/task9_evaluation.yaml \
  --rerun-complete \
  --log-every 5000
' > logs/task9/lightweight_sam_evaluation.log 2>&1 &

echo $! > logs/task9/lightweight_sam_evaluation.pid
tail -f logs/task9/lightweight_sam_evaluation.log
```

## Speed Benchmark

```bash
nohup bash -lc '
set -euo pipefail
export PYTHONUNBUFFERED=1

python scripts/benchmarks/measure_task7_inference_speed.py \
  --config configs/task9_inference_speed.yaml \
  --rerun-complete
' > logs/task9/lightweight_sam_speed.log 2>&1 &

echo $! > logs/task9/lightweight_sam_speed.pid
tail -f logs/task9/lightweight_sam_speed.log
```

## Final Summary

After inference, evaluation, and speed runs finish:

```bash
python scripts/analysis/summarize_task9_lightweight_sam.py

cat outputs/task9_lightweight_sam/summary/summary.json
```

The final recommendation should be based on mIoU, boundary F1, mask AP, FPS,
CPU latency, GPU latency, and checkpoint size together. A model is considered
edge-feasible only if its quality loss is acceptable for the robotic challenge
category and its measured latency is compatible with the target robot loop.
