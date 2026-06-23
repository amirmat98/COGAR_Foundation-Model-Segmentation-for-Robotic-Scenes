# Report Alignment

Project: **Foundation Model Segmentation for Robotic Scenes**  
Assignment: **Zero-Shot Segmentation Benchmark for Robotic Perception (Simulation)**  
Student id: **5884715**

The repository is organized around two top-level deliverables:

| File | Role |
| --- | --- |
| `README.md` | GitHub-facing technical guide for datasets, scripts, configs, artifacts, and reproduction. |
| `REPORT.md` | Final research report following the required lecture structure. |

All files under `report/` are supporting material for the root `REPORT.md`.
They contain section drafts, slide planning, source mapping, and wording notes.
They are not separate final deliverables.

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
| `report/references.md` | Detailed bibliography and source map. |

## Evidence Sources

The final report draws evidence from:

- `docs/tasks/` for task-level implementation notes,
- `configs/` for dataset/model/evaluation definitions,
- `outputs/final_benchmark_assets/` for plots and compact result tables,
- `outputs/task*_*/` for intermediate benchmark artifacts.

Raw checkpoints and prediction JSONL files are kept under `results/` when
available, but they are not committed to Git because of size.
