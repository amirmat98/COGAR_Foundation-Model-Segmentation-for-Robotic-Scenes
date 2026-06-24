# Report Alignment

Project: **Foundation Model Segmentation for Robotic Scenes**  
Assignment: **Zero-Shot Segmentation Benchmark for Robotic Perception (Simulation)**  
Student id: **5884715**

The repository is organized around two top-level deliverables:

| File | Role |
| --- | --- |
| `README.md` | GitHub-facing technical guide for datasets, scripts, configs, artifacts, and reproduction. |
| `REPORT.md` | Final research report following the required lecture structure. |

Files under `report/` are supporting material for the root `REPORT.md`. They
are concise companion pages, not duplicated final reports. Reusable figures and
CSV evidence are centralized in `report/figures_and_tables.md` to avoid
repeating the same plots in every section file.

> **Storage note:** The complete `results/` folder could not be included in Git
> because its raw predictions and checkpoints are too large. Git contains the
> compact evidence under `outputs/`; full raw results remain on the benchmark
> machine/AWS storage.

`REPORT.md` now acts as the hub: each required report section includes a direct
link to the relevant detailed supporting file below. The supporting files are
therefore dependencies for explanation depth, while `REPORT.md` remains the
single final research report.

## Required Research Structure

The final report and presentation are organized around:

1. Research Problem
2. State of the Art
3. Research Formulation
4. Cognitive Approach
5. Congruence of Results and Conclusions

## Supporting Files

| Supporting file | Purpose |
| --- | --- |
| `report/00_presentation_report_roadmap.md` | Presentation/report roadmap. |
| `report/01_research_problem.md` | Expanded research-problem material. |
| `report/02_state_of_the_art.md` | Literature and model-family background. |
| `report/03_research_formulation.md` | Research question, objectives, hypotheses, and methodology. |
| `report/04_cognitive_approach.md` | COGAR connection and cognitive interpretation. |
| `report/05_results_congruence_and_conclusions.md` | Evidence-to-conclusion discipline. |
| `report/06_slide_deck_outline.md` | Slide-by-slide presentation outline. |
| `report/figures_and_tables.md` | Shared catalog of plots, summary tables, and representative failure images. |
| `report/references.md` | Detailed bibliography and source map. |

## Repository-Local Wiki

The folder `docs/wiki/` provides a GitHub-wiki-style version of the report:

- `docs/wiki/README.md`
- `docs/wiki/Research-Problem.md`
- `docs/wiki/State-of-the-Art.md`
- `docs/wiki/Methodology.md`
- `docs/wiki/Cognitive-Approach.md`
- `docs/wiki/Results-and-Recommendations.md`
- `docs/wiki/Artifacts.md`

These files can be kept in the repository or copied into GitHub's separate wiki
repository if a hosted Wiki tab is required.

## Evidence Sources

The final report draws evidence from:

- `docs/tasks/` for task-level implementation notes,
- `configs/` for dataset/model/evaluation definitions,
- `outputs/final_benchmark_assets/` for plots and compact result tables,
- `outputs/task*_*/` for intermediate benchmark artifacts.

Raw checkpoints and prediction JSONL files are kept under `results/` on the
benchmark machine, but the complete folder could not be committed to Git
because it is too large.
