# Local Data Directory

This directory is reserved for COGAR-SimRobotics-500 data and related local
dataset artifacts.

The final benchmark dataset is documented as:

```text
data/cogar_sim_500_final/
  rgb/
  annotations/
    sim_robotic_scenes_index_final_filtered.csv
  instance_masks/
  semantic_masks/
  binary_masks/
  metadata/
  splits/
```

Final dataset summary:

| Property | Value |
|---|---:|
| Clean images | 500 |
| Object instances | 4,471 |
| Categories | 9 |
| Challenge groups | 5 |
| Main use | zero-shot and supervised instance-segmentation benchmark |

Large generated RGB images, masks, depth maps, raw simulation exports, and
external datasets should stay local or in external storage. Lightweight
annotation files, indexes, schemas, and documentation may be committed when they
are needed to reproduce the final benchmark tables.

Supervised-baseline conversion artifacts may also exist locally, for example:

```text
data/yolo_cogar_sim_500_final/
```

Do not place model weights or checkpoints in `data/`. Keep them in
`checkpoints/` or external storage, where checkpoint extensions are ignored by
Git.
