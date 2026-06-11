param(
    [ValidateSet("help", "check", "index", "box", "point", "auto-fast16", "report", "all-sam-fast")]
    [string]$Job = "help",

    [string]$OcidRoot = $(if ($env:OCID_ROOT) { $env:OCID_ROOT } else { "C:\COGAR_DATASETs\OCID-dataset" }),
    [string]$Config = $(if ($env:CONFIG) { $env:CONFIG } else { "configs\paths.yaml" }),
    [string]$Index = $(if ($env:INDEX) { $env:INDEX } else { "outputs\ocid_full\indexes\ocid_full_objects_filtered_with_masks.csv" }),
    [string]$Checkpoint = $(if ($env:CHECKPOINT) { $env:CHECKPOINT } else { "checkpoints\sam_vit_b_01ec64.pth" }),
    [string]$ModelType = $(if ($env:MODEL_TYPE) { $env:MODEL_TYPE } else { "vit_b" }),
    [string]$Device = $(if ($env:DEVICE) { $env:DEVICE } else { "cuda" }),
    [int]$ProgressEvery = $(if ($env:PROGRESS_EVERY) { [int]$env:PROGRESS_EVERY } else { 500 }),
    [bool]$NoSaveMasks = $(if ($env:NO_SAVE_MASKS) { [bool]::Parse($env:NO_SAVE_MASKS) } else { $true }),

    [int]$AutoPointsPerSide = $(if ($env:AUTO_POINTS_PER_SIDE) { [int]$env:AUTO_POINTS_PER_SIDE } else { 16 }),
    [double]$AutoPredIouThresh = $(if ($env:AUTO_PRED_IOU_THRESH) { [double]$env:AUTO_PRED_IOU_THRESH } else { 0.90 }),
    [double]$AutoStabilityScoreThresh = $(if ($env:AUTO_STABILITY_SCORE_THRESH) { [double]$env:AUTO_STABILITY_SCORE_THRESH } else { 0.92 }),
    [int]$AutoCropNLayers = $(if ($env:AUTO_CROP_N_LAYERS) { [int]$env:AUTO_CROP_N_LAYERS } else { 0 }),

    [string]$BoxLimit = $(if ($env:BOX_LIMIT) { $env:BOX_LIMIT } else { "" }),
    [string]$PointLimit = $(if ($env:POINT_LIMIT) { $env:POINT_LIMIT } else { "" }),
    [string]$AutoLimit = $(if ($env:AUTO_LIMIT) { $env:AUTO_LIMIT } else { "" })
)

$ErrorActionPreference = "Stop"

function Show-Usage {
    @"
Run OCID benchmark jobs on an Azure Windows GPU VM.

Usage:
  .\scripts\azure\run_ocid_azure.ps1 check
  .\scripts\azure\run_ocid_azure.ps1 index
  .\scripts\azure\run_ocid_azure.ps1 box
  .\scripts\azure\run_ocid_azure.ps1 point
  .\scripts\azure\run_ocid_azure.ps1 auto-fast16
  .\scripts\azure\run_ocid_azure.ps1 report
  .\scripts\azure\run_ocid_azure.ps1 all-sam-fast

Useful environment variables:
  `$env:OCID_ROOT = "D:\data\OCID-dataset"
  `$env:CHECKPOINT = "checkpoints\sam_vit_b_01ec64.pth"
  `$env:DEVICE = "cuda"
  `$env:NO_SAVE_MASKS = "true"
  `$env:AUTO_POINTS_PER_SIDE = "16"
  `$env:BOX_LIMIT = "500"
  `$env:POINT_LIMIT = "500"
  `$env:AUTO_LIMIT = "1000"
"@
}

function Invoke-Step {
    param([string[]]$Command)

    Write-Host ""
    Write-Host "[AZURE] $($Command -join ' ')"
    $exe = $Command[0]
    $args = @()
    if ($Command.Count -gt 1) {
        $args = $Command[1..($Command.Count - 1)]
    }
    & $exe @args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $($Command -join ' ')"
    }
}

function Require-Path {
    param([string]$PathValue)
    if (-not (Test-Path $PathValue)) {
        throw "Missing required path: $PathValue"
    }
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot
$env:PYTHONPATH = "$RepoRoot\src;$env:PYTHONPATH"

$BoxOutputDir = if ($env:BOX_OUTPUT_DIR) { $env:BOX_OUTPUT_DIR } else { "outputs\ocid_full\azure\sam_vit_b_box" }
$PointOutputDir = if ($env:POINT_OUTPUT_DIR) { $env:POINT_OUTPUT_DIR } else { "outputs\ocid_full\azure\sam_vit_b_point" }
$AutoOutputDir = if ($env:AUTO_OUTPUT_DIR) { $env:AUTO_OUTPUT_DIR } else { "outputs\ocid_full\azure\sam_vit_b_auto_fast16" }

$BoxResults = if ($env:BOX_RESULTS) { $env:BOX_RESULTS } else { "outputs\ocid_full\results\sam_vit_b_box.csv" }
$PointResults = if ($env:POINT_RESULTS) { $env:POINT_RESULTS } else { "outputs\ocid_full\results\sam_vit_b_point.csv" }
$AutoFastResults = if ($env:AUTO_FAST_RESULTS) { $env:AUTO_FAST_RESULTS } else { "outputs\ocid_full\results\sam_vit_b_auto_fast16.csv" }

function Print-Environment {
    Write-Host "[AZURE] Repo: $RepoRoot"
    Write-Host "[AZURE] OCID_ROOT: $OcidRoot"
    Write-Host "[AZURE] CONFIG: $Config"
    Write-Host "[AZURE] INDEX: $Index"
    Write-Host "[AZURE] CHECKPOINT: $Checkpoint"
    Write-Host "[AZURE] MODEL_TYPE: $ModelType"
    Write-Host "[AZURE] DEVICE: $Device"
    Write-Host "[AZURE] PROGRESS_EVERY: $ProgressEvery"
    Write-Host "[AZURE] NO_SAVE_MASKS: $NoSaveMasks"
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        nvidia-smi
    } else {
        Write-Host "[WARN] nvidia-smi not found"
    }
}

function Ensure-Index {
    if ((Test-Path $Index) -and ($env:FORCE_INDEX -ne "1")) {
        Write-Host "[AZURE] Reusing existing OCID index: $Index"
        return
    }

    Require-Path $Config
    Require-Path $OcidRoot
    Invoke-Step @(
        "python", "scripts\dataset\create_object_index.py",
        "--dataset", "ocid_full",
        "--config", $Config,
        "--ocid-root", $OcidRoot,
        "--progress-every", "250"
    )
}

function Mask-Args {
    if ($NoSaveMasks) {
        return @("--no-save-masks")
    }
    return @()
}

function Run-Box {
    Ensure-Index
    Require-Path $Checkpoint
    $cmd = @(
        "python", "scripts\eval\run_sam_box_prompt.py",
        "--config", $Config,
        "--index", $Index,
        "--checkpoint", $Checkpoint,
        "--model-type", $ModelType,
        "--device", $Device,
        "--output-dir", $BoxOutputDir,
        "--results-csv", $BoxResults,
        "--no-visualizations",
        "--progress-every", "$ProgressEvery"
    ) + (Mask-Args)
    if ($BoxLimit) {
        $cmd += @("--limit", $BoxLimit)
    }
    Invoke-Step $cmd
}

function Run-Point {
    Ensure-Index
    Require-Path $Checkpoint
    $cmd = @(
        "python", "scripts\eval\run_sam_point_prompt.py",
        "--config", $Config,
        "--index", $Index,
        "--checkpoint", $Checkpoint,
        "--model-type", $ModelType,
        "--device", $Device,
        "--output-dir", $PointOutputDir,
        "--results-csv", $PointResults,
        "--no-visualizations",
        "--progress-every", "$ProgressEvery"
    ) + (Mask-Args)
    if ($PointLimit) {
        $cmd += @("--limit", $PointLimit)
    }
    Invoke-Step $cmd
}

function Run-AutoFast16 {
    Ensure-Index
    Require-Path $Checkpoint
    $cmd = @(
        "python", "scripts\eval\run_sam_auto_masks.py",
        "--config", $Config,
        "--index", $Index,
        "--checkpoint", $Checkpoint,
        "--model-type", $ModelType,
        "--device", $Device,
        "--output-dir", $AutoOutputDir,
        "--results-csv", $AutoFastResults,
        "--points-per-side", "$AutoPointsPerSide",
        "--pred-iou-thresh", "$AutoPredIouThresh",
        "--stability-score-thresh", "$AutoStabilityScoreThresh",
        "--crop-n-layers", "$AutoCropNLayers",
        "--progress-every", "$ProgressEvery"
    )
    if ($NoSaveMasks) {
        $cmd += "--no-save-masks"
    }
    if ($AutoLimit) {
        $cmd += @("--limit", $AutoLimit)
    }
    Invoke-Step $cmd
}

function Run-Report {
    Ensure-Index
    $resultArgs = @()
    foreach ($csv in @($BoxResults, $PointResults, $AutoFastResults)) {
        if ((Test-Path $csv) -and ((Get-Item $csv).Length -gt 0)) {
            $resultArgs += $csv
        } else {
            Write-Host "[AZURE] Report skip missing/empty result: $csv"
        }
    }
    $cmd = @("python", "scripts\analysis\summarize_ocid_massive_benchmark.py", "--debug")
    if ($resultArgs.Count -gt 0) {
        $cmd += "--results"
        $cmd += $resultArgs
    }
    Invoke-Step $cmd
}

switch ($Job) {
    "help" {
        Show-Usage
    }
    "check" {
        Print-Environment
        Require-Path $Config
        Require-Path $OcidRoot
        Require-Path $Checkpoint
        Invoke-Step @(
            "python", "-c",
            "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
        )
    }
    "index" {
        Print-Environment
        $env:FORCE_INDEX = if ($env:FORCE_INDEX) { $env:FORCE_INDEX } else { "1" }
        Ensure-Index
    }
    "box" {
        Print-Environment
        Run-Box
    }
    "point" {
        Print-Environment
        Run-Point
    }
    "auto-fast16" {
        Print-Environment
        Run-AutoFast16
    }
    "report" {
        Print-Environment
        Run-Report
    }
    "all-sam-fast" {
        Print-Environment
        Ensure-Index
        Run-Box
        Run-Point
        Run-AutoFast16
        Run-Report
    }
}
