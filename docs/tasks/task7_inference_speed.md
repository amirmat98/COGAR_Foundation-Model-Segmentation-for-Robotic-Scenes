# Task 7 - Inference Speed Benchmark

Task 7 measures segmentation inference speed on GPU and CPU for the models used in Tasks 4 and 5.

## Scope

The benchmark covers:

- SAM ViT-H and SAM ViT-B with point, box, and automatic mask generation modes.
- SAM2 Hiera-Large with point, box, and automatic mask generation modes.
- FastSAM-X with point, box, and automatic mask generation modes.
- YOLOv8-seg, Mask R-CNN, and DeepLabV3+ supervised baselines from Task 5.

The script records FPS, mean latency, median latency, P95 latency, min/max latency, sample count, device metadata, and output count statistics.

## Protocol

The benchmark uses the Task 4 prompt manifests to select one representative prompt/image for point and box modes. Automatic mode and supervised baselines are measured per image.

Disk image loading is excluded from the timed loop where possible. Model preprocessing, model inference, and mask/post-processing needed to produce segmentation output are included. CUDA timings are synchronized before and after each measured image.

CPU runs use smaller sample counts than GPU runs. This is intentional because large SAM automatic-mask generation is not expected to be real-time on CPU; the sample count is stored in every output row.

## Outputs

Compact outputs are written to:

```text
outputs/task7_inference_speed/
```

The main files are:

```text
outputs/task7_inference_speed/summary.csv
outputs/task7_inference_speed/summary.json
```

Per-run JSON files are stored under:

```text
outputs/task7_inference_speed/<device>/<dataset>/<model>/<mode>_speed.json
```

## Commands

Run setup checks:

```bash
cd ~/COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes
source .venv/bin/activate

python -m py_compile scripts/benchmarks/measure_task7_inference_speed.py

python scripts/benchmarks/measure_task7_inference_speed.py \
  --datasets blenderproc_cogar_sim \
  --models sam_vit_b yolo8_seg \
  --prompt-modes point automatic \
  --devices cuda cpu \
  --max-images 2 \
  --automatic-max-images 1 \
  --cpu-max-images 1 \
  --cpu-automatic-max-images 1 \
  --dry-run
```

Run a GPU smoke test:

```bash
mkdir -p logs/task7

nohup bash -lc '
set -euo pipefail
export PYTHONUNBUFFERED=1

python scripts/benchmarks/measure_task7_inference_speed.py \
  --datasets blenderproc_cogar_sim \
  --models sam_vit_b fastsam_x yolo8_seg mask_rcnn deeplabv3plus \
  --prompt-modes point automatic \
  --devices cuda \
  --max-images 3 \
  --automatic-max-images 2 \
  --warmup-images 1 \
  --rerun-complete \
  --log-every 1
' > logs/task7/speed_smoke_gpu_blenderproc.log 2>&1 &

echo $! > logs/task7/speed_smoke_gpu_blenderproc.pid
tail -f logs/task7/speed_smoke_gpu_blenderproc.log
```

Run a CPU smoke test:

```bash
nohup bash -lc '
set -euo pipefail
export PYTHONUNBUFFERED=1

python scripts/benchmarks/measure_task7_inference_speed.py \
  --datasets blenderproc_cogar_sim \
  --models sam_vit_b yolo8_seg deeplabv3plus \
  --prompt-modes point \
  --devices cpu \
  --cpu-max-images 2 \
  --cpu-automatic-max-images 1 \
  --warmup-images 1 \
  --rerun-complete \
  --log-every 1
' > logs/task7/speed_smoke_cpu_blenderproc.log 2>&1 &

echo $! > logs/task7/speed_smoke_cpu_blenderproc.pid
tail -f logs/task7/speed_smoke_cpu_blenderproc.log
```

Run the full Task 7 benchmark:

```bash
nohup bash -lc '
set -euo pipefail
export PYTHONUNBUFFERED=1

python scripts/benchmarks/measure_task7_inference_speed.py \
  --devices cuda cpu \
  --max-images 50 \
  --automatic-max-images 20 \
  --cpu-max-images 10 \
  --cpu-automatic-max-images 1 \
  --warmup-images 3 \
  --rerun-complete \
  --log-every 5
' > logs/task7/speed_full_all_models_all_datasets.log 2>&1 &

echo $! > logs/task7/speed_full_all_models_all_datasets.pid
tail -f logs/task7/speed_full_all_models_all_datasets.log
```

After completion:

```bash
cat outputs/task7_inference_speed/summary.csv
find outputs/task7_inference_speed -name '*_speed.json' | wc -l
du -sh outputs/task7_inference_speed logs/task7
```

Commit after syncing the compact outputs and logs:

```bash
git add outputs/task7_inference_speed logs/task7/*.log
git commit -m "record Task 7 inference speed metrics"
git push
```
