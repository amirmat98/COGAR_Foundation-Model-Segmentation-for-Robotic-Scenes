# Local Outputs Directory

This directory is reserved for benchmark outputs and final result evidence.

For the final assignment submission, lightweight outputs may be committed when
they make the benchmark reproducible or auditable, including:

- final CSV/JSON result tables
- summary tables used by the Markdown reports
- compact plots and qualitative figures
- baseline training summaries
- logs that document final completed runs

Heavy or temporary artifacts should remain ignored or local:

- predicted mask dumps
- raw generated images
- model weights and checkpoints
- large temporary experiment folders
- scratch reruns that are not part of the final evidence

The output folders document evidence for the final benchmark, including
foundation-model evaluations, supervised baselines, metrics, speed summaries,
and failure-mode analysis. The canonical human-readable interpretation remains
in `README.md` and `docs/`.

Final report figures are generated into:

```text
outputs/figures/final_report/
```

These figures are lightweight PNG charts and montages referenced by the final
Markdown reports.
