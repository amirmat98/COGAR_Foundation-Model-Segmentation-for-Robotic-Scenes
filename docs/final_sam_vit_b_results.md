
# Final SAM ViT-B Results on COGAR-SimRobotics-500

## Dataset

- Final dataset: COGAR-SimRobotics-500
- Clean images: 500
- Object instances evaluated: 4,471
- Final index: `data/cogar_sim_500_final/annotations/sim_robotic_scenes_index_final_filtered.csv`
- Bad images after final filtering: 0

## Visual evidence

![SAM ViT-B prompt context in mean IoU chart](/outputs/figures/final_report/metrics/mean_iou_by_model_prompt.png)

*Figure: SAM ViT-B box, point, and auto results shown in the full zero-shot prompt-mode comparison.*

![SAM ViT-B boundary F1 context](/outputs/figures/final_report/metrics/boundary_f1_by_model_prompt.png)

*Figure: Boundary F1 comparison showing SAM ViT-B prompt-mode boundary quality relative to other evaluated models.*

## SAM ViT-B prompt comparison

| Prompt | Objects | Mean IoU | Median IoU | Boundary F1 | IoU >= 0.90 | IoU >= 0.75 | IoU >= 0.50 | IoU < 0.10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Box | 4,471 | 0.9057 | 0.9553 | 0.9356 | 0.7513 | 0.9137 | 0.9703 | 0.0031 |
| Point | 4,471 | 0.7985 | 0.9125 | 0.8131 | 0.5278 | 0.7271 | 0.8662 | 0.0204 |
| Auto | 4,471 | 0.8025 | 0.9422 | 0.8381 | 0.6046 | 0.7486 | 0.8678 | 0.0552 |

## Interpretation

SAM ViT-B with box prompts achieved the best overall performance on the final simulated robotic-scene dataset. It reached 0.9057 mean IoU, 0.9553 median IoU, and 0.9356 mean Boundary F1. It also had the lowest catastrophic failure rate, with only 0.31% of evaluated objects below IoU 0.10.

Point prompting was less reliable than box prompting. Although the median IoU remained high at 0.9125, the mean IoU dropped to 0.7985, indicating more unstable behavior on difficult objects such as small parts, thin cables, transparent objects, and occluded instances.

Automatic mask generation achieved a high median IoU of 0.9422, but its catastrophic failure rate was 5.52%, much higher than box or point prompting. This suggests that automatic full-image proposals often segment easy objects well but can miss individual target objects, which is important for robotic perception where missed objects can cause task failures.

Overall, box prompting is the recommended SAM ViT-B mode for this benchmark when object proposals or bounding boxes are available. Automatic mask generation is useful for open-world scene exploration but is less reliable for guaranteed per-object segmentation.
