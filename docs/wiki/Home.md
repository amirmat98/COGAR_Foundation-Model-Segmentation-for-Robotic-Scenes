# Project Wiki

This folder is a repository-local wiki for the benchmark. It mirrors the final
report structure but keeps the pages short and navigable.

> **Result storage notice:** The complete `results/` folder could not be
> included in Git because its raw predictions and checkpoints are too large.
> Compact evaluated summaries, plots, and tables are available under
> `outputs/`; full raw results remain on the benchmark machine/AWS storage.

Main documents:

| Page | Purpose |
|---|---|
| [Research Problem](Research-Problem.md) | What problem the benchmark addresses and why robotics makes it difficult. |
| [State of the Art](State-of-the-Art.md) | Background on foundation segmentation, lightweight variants, baselines, and simulation. |
| [Methodology](Methodology.md) | Dataset/model/prompt/metric design and fairness rules. |
| [Cognitive Approach](Cognitive-Approach.md) | COGAR interpretation of segmentation as a perception module. |
| [Results and Recommendations](Results-and-Recommendations.md) | Main numerical results, trade-offs, and final recommendations. |
| [Artifacts](Artifacts.md) | Where to find plots, tables, reports, configs, and task outputs. |

Primary entry points:

- Final report: [../../REPORT.md](../../REPORT.md)
- Technical repository guide: [../../README.md](../../README.md)
- Report support files: [../../report/](../../report/)
- Figure/table catalog: [../../report/figures_and_tables.md](../../report/figures_and_tables.md)

To use this as a GitHub Wiki, copy these Markdown files into the repository's
separate GitHub Wiki repository, or keep this folder as the in-repo wiki.
