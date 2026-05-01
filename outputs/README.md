# Local Outputs Directory

This directory is reserved for generated experiment outputs.

Do not commit generated masks, visualizations, benchmark CSVs, plots, logs,
or model outputs. The OCID debug pipeline currently writes files such as:

- `outputs/indexes/`
- `outputs/gt_binary_masks/`
- `outputs/sam_box_prompt/`
- `outputs/sam_point_prompt_batch/`
- `outputs/analysis/`

Important results should be summarized in `README.md`, `docs/`, or `reports/`
rather than committed as generated artifacts.
