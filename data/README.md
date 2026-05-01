# Local Data Directory

This directory is reserved for local datasets and generated dataset artifacts.

Large data should not be committed to Git. The current project stages use:

- `datasets/OCID-dataset/` for the local OCID debug subset.
- `data/cogar_sim_500/` for the planned simulated robotic-scene dataset.
- `sim_dataset/` only for temporary or pilot generated samples.

Keep generated images, masks, depth maps, annotations, and external dataset
exports in ignored local folders or external storage. Commit only small schema,
configuration, template, and documentation files.
