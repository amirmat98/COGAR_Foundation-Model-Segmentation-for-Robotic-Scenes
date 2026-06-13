# Public Dataset Release

Use one public URL per generated dataset and record it in
`configs/datasets.yaml`.

Recommended:

1. Zenodo for final release and DOI.
2. Hugging Face Datasets for ML-friendly dataset hosting.
3. Google Drive or university storage for simple sharing.

For this project, release these generated datasets:

- Isaac official Unitree G1
- BlenderProc COGAR-SimRobotics-1000

Do not re-host OCID unless its license allows redistribution. Link to the
upstream OCID source instead.

Prepare a release archive:

```bash
.venv/bin/python scripts/datasets/package_dataset_release.py \
  /mnt/Info/COGAR_DATASETs/Isacc_dataset/datasets/robotic_sdg_v3_official_g1_1000 \
  --name robotic_sdg_v3_official_g1_1000
```

```bash
.venv/bin/python scripts/datasets/package_dataset_release.py \
  /mnt/Info/COGAR_DATASETs/BlenderProc_cogar_sim_1000 \
  --name BlenderProc_cogar_sim_1000
```

Then upload the archive and checksum file to the chosen hosting service.

Useful official documentation:

- Zenodo upload guide: https://help.zenodo.org/docs/deposit/create-new-upload/
- Hugging Face dataset upload guide: https://huggingface.co/docs/hub/en/datasets-adding
- Google Drive sharing guide: https://support.google.com/drive/answer/2494822
