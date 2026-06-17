# Public Dataset Release

This project publishes only the generated datasets:

- Isaac official Unitree G1 synthetic dataset
- BlenderProc COGAR-SimRobotics-1000 synthetic dataset

![Generated and external dataset examples](../../outputs/final_benchmark_assets/plots/dataset_examples.png)

OCID is an external dataset. Do not re-host OCID unless its upstream license
explicitly permits redistribution. Link to the upstream OCID project page
instead: https://www.acin.tuwien.ac.at/object-clutter-indoor-dataset/

## Published Release

The generated Isaac and BlenderProc datasets are published in one Zenodo
record:

```text
https://doi.org/10.5281/zenodo.20736993
```

Record page:

```text
https://zenodo.org/records/20736993
```

All-versions DOI:

```text
https://doi.org/10.5281/zenodo.20736992
```

## Recommended Host

Zenodo was used for the final public release because it provides a citable DOI
and supports large dataset deposits. Zenodo's current documentation allows up
to 100 files and 50 GB per upload, which is enough for these two archives.

Official references:

- Zenodo create-new-upload guide: https://help.zenodo.org/docs/deposit/create-new-upload/
- Zenodo file limits and file-management guide: https://help.zenodo.org/docs/deposit/manage-files/

## Release Files

Package the datasets outside the Git repository:

```text
/mnt/Info/COGAR_DATASETs/releases/
```

Upload these files to the same Zenodo dataset record:

```text
robotic_sdg_v3_official_g1_1000.tar.gz
robotic_sdg_v3_official_g1_1000.tar.gz.sha256
robotic_sdg_v3_official_g1_1000_release_manifest.json
BlenderProc_cogar_sim_1000.tar.gz
BlenderProc_cogar_sim_1000.tar.gz.sha256
BlenderProc_cogar_sim_1000_release_manifest.json
```

## Package Locally

First run a dry run:

```bash
.venv/bin/python scripts/datasets/package_dataset_release.py \
  /mnt/Info/COGAR_DATASETs/Isacc_dataset/datasets/robotic_sdg_v3_official_g1_1000 \
  --name robotic_sdg_v3_official_g1_1000 \
  --dry-run

.venv/bin/python scripts/datasets/package_dataset_release.py \
  /mnt/Info/COGAR_DATASETs/BlenderProc_cogar_sim_1000 \
  --name BlenderProc_cogar_sim_1000 \
  --dry-run
```

Then run the full packaging job with progress logging:

```bash
mkdir -p logs/release /mnt/Info/COGAR_DATASETs/releases

nohup bash -lc '
set -euo pipefail
export PYTHONUNBUFFERED=1

.venv/bin/python scripts/datasets/package_dataset_release.py \
  /mnt/Info/COGAR_DATASETs/Isacc_dataset/datasets/robotic_sdg_v3_official_g1_1000 \
  --name robotic_sdg_v3_official_g1_1000 \
  --force

.venv/bin/python scripts/datasets/package_dataset_release.py \
  /mnt/Info/COGAR_DATASETs/BlenderProc_cogar_sim_1000 \
  --name BlenderProc_cogar_sim_1000 \
  --force
' > logs/release/package_dataset_release.log 2>&1 &

echo $! > logs/release/package_dataset_release.pid
tail -f logs/release/package_dataset_release.log
```

Verify checksums before upload:

```bash
cd /mnt/Info/COGAR_DATASETs/releases
sha256sum -c robotic_sdg_v3_official_g1_1000.tar.gz.sha256
sha256sum -c BlenderProc_cogar_sim_1000.tar.gz.sha256
```

## Zenodo Metadata

Use one Zenodo record for both generated datasets.

- Resource type: Dataset
- Title: COGAR Simulated Robotic Scene Segmentation Datasets
- Creators: use the project author name and affiliation required for submission
- Description: Synthetic robotic-scene segmentation datasets generated for the
  Foundation Model Segmentation for Robotic Scenes benchmark. The release
  contains an Isaac Sim dataset using the official Unitree G1 asset and a
  BlenderProc tabletop robotics dataset, with RGB images, segmentation masks,
  COCO annotations, metadata, and checksums.
- Keywords: robotic perception, segmentation, synthetic data, Isaac Sim,
  BlenderProc, Unitree G1, SAM benchmark
- Related identifiers: link this GitHub repository and the project report
- License: choose a license approved for the generated data and source-asset
  terms. Do not include upstream USD/source assets if their license is separate;
  the release archives are intended for rendered images, annotations, and
  metadata.

## Documentation Update

The public DOI is recorded in:

- `configs/datasets.yaml`
- `docs/datasets/isaac_official_g1.md`
- `docs/datasets/blenderproc_cogar_sim_1000.md`
- `REPORT.md`

Suggested commit message for this documentation update:

```text
record public dataset release links
```
