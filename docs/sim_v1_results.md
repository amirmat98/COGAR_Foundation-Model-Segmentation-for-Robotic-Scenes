# COGAR-Sim v1-clean-small Results

## Dataset quality

- Images: 15
- Object instances: 271
- Bad images after audit: 0
- Objects per image: min 12, median 18, mean 18.07, max 25
- Table/support category excluded from benchmark index
- Object-level flags corrected:
  - reflective: metal_part, tool
  - transparent: glass_object
  - small_part: screw, connector
  - dynamic: dynamic_scene

## SAM ViT-B Results

| Prompt | Objects | Mean IoU | Median IoU | Boundary F1 | IoU >= 0.90 | IoU >= 0.75 | IoU >= 0.50 | IoU < 0.10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Box | 271 | 0.8931 | 0.9514 | 0.9296 | 0.7011 | 0.9041 | 0.9668 | 0.0074 |
| Point | 271 | 0.8133 | 0.9228 | 0.8295 | 0.5498 | 0.7712 | 0.8708 | 0.0185 |
| Auto | 271 | 0.7751 | 0.9386 | 0.8185 | 0.6089 | 0.7638 | 0.8266 | 0.1033 |

## Interpretation

Box prompting is the most reliable prompt mode on this dataset, with the highest mean IoU and lowest catastrophic failure rate. Point prompting is usable but less stable. Automatic mask generation has a high median IoU but a much higher catastrophic failure rate, meaning it often works well on easy objects but misses some instances entirely.

This v1 dataset is suitable as a validation dataset, not as the final benchmark dataset, because it contains only 15 images and still has limited representation of small-parts and transparent-glass cases.
