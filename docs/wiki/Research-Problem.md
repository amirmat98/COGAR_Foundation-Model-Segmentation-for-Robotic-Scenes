# Research Problem

Robotic scene segmentation is not only an image-processing problem. A robot
uses object masks to support manipulation, tracking, inspection, navigation,
planning, and safety checks.

The benchmark asks:

> To what extent can promptable foundation segmentation models provide reliable
> zero-shot object masks for robotic scene understanding in challenging
> simulated environments, and what trade-offs appear against lightweight and
> supervised alternatives in accuracy, robustness, prompt dependence, and
> real-time feasibility?

Key robotic difficulties:

- transparent glass and plastic,
- reflective metal,
- partial occlusion,
- small screws, cables, connectors, and tools,
- robot-body visibility,
- dynamic or moving objects,
- real-time inference constraints.

Research subquestions:

| ID | Question |
|---|---|
| RQ1 | Which SAM-family models produce the best masks under point, box, and automatic prompting? |
| RQ2 | Which models fail under robotic challenge groups? |
| RQ3 | Which models are fast enough for plausible robotic use? |
| RQ4 | When should a robot use heavy SAM, lightweight SAM, or a supervised baseline? |

More detail:

- [../../report/01_research_problem.md](../../report/01_research_problem.md)
- [../../REPORT.md#1-research-problem](../../REPORT.md#1-research-problem)
