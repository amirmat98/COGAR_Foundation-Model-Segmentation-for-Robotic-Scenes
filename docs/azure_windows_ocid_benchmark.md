# Azure Windows OCID Benchmark Runbook

Use this when the benchmark is running on an Azure Windows GPU VM instead of a
Linux EC2 instance.

## 1. Recommended Shell

Open PowerShell in the repository root. If script execution is blocked for this
terminal session:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies if needed:

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt -r requirements-models.txt
```

For CUDA, install a CUDA-enabled PyTorch build that matches the NVIDIA driver on
the VM. Verify before starting the benchmark:

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## 2. Put Data In A Windows Path

Use a simple local path such as:

```text
D:\data\OCID-dataset
```

Then set:

```powershell
$env:OCID_ROOT = "D:\data\OCID-dataset"
```

The scripts also work with:

```powershell
$env:OCID_ROOT = "C:\COGAR_DATASETs\OCID-dataset"
```

## 3. Check The VM

```powershell
.\scripts\azure\run_ocid_azure.ps1 check
```

This checks:

- OCID path
- checkpoint path
- PyTorch import
- CUDA availability
- visible GPU name

## 4. Build The OCID Index

```powershell
.\scripts\azure\run_ocid_azure.ps1 index
```

Expected index output:

```text
outputs\ocid_full\indexes\ocid_full_objects_filtered_with_masks.csv
```

## 5. Pilot Run

Use limits first:

```powershell
$env:BOX_LIMIT = "500"
$env:POINT_LIMIT = "500"
$env:AUTO_LIMIT = "1000"
.\scripts\azure\run_ocid_azure.ps1 all-sam-fast
```

Clear the limits before the full run:

```powershell
Remove-Item Env:\BOX_LIMIT -ErrorAction SilentlyContinue
Remove-Item Env:\POINT_LIMIT -ErrorAction SilentlyContinue
Remove-Item Env:\AUTO_LIMIT -ErrorAction SilentlyContinue
```

## 6. Full Run

Run all main SAM ViT-B OCID jobs:

```powershell
.\scripts\azure\run_ocid_azure.ps1 all-sam-fast
```

This runs:

- SAM ViT-B box prompts
- SAM ViT-B point prompts
- SAM ViT-B automatic masks with faster `points_per_side=16`
- OCID report generation

Default speed/disk settings:

```text
NO_SAVE_MASKS=true
AUTO_POINTS_PER_SIDE=16
AUTO_PRED_IOU_THRESH=0.90
AUTO_STABILITY_SCORE_THRESH=0.92
PROGRESS_EVERY=500
```

To change them:

```powershell
$env:NO_SAVE_MASKS = "false"
$env:AUTO_POINTS_PER_SIDE = "32"
$env:PROGRESS_EVERY = "100"
```

## 7. Individual Jobs

```powershell
.\scripts\azure\run_ocid_azure.ps1 box
.\scripts\azure\run_ocid_azure.ps1 point
.\scripts\azure\run_ocid_azure.ps1 auto-fast16
.\scripts\azure\run_ocid_azure.ps1 report
```

## 8. Outputs

```text
outputs\ocid_full\results\
outputs\ocid_full\tables\
docs\ocid_massive_benchmark_report.md
```

Copy those folders back from the VM after the run.
