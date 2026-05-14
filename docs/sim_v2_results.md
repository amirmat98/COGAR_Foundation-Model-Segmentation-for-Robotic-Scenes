# COGAR-Sim v2-clean-medium Results

## Dataset quality

- Raw generated images: 120
- Valid images before filtering: 88
- Clean images after filtering: 87
- Object instances after filtering: 764
- Bad images removed: 1 (`000101.png`)
- Objects per image: min 1, median 9.0, mean 8.69, max 15
- Table/support objects excluded from the benchmark index

## Dataset distribution

### Challenge counts

| Challenge | Objects |
|---|---:|
| small_parts | 202 |
| dynamic_scene | 162 |
| partial_occlusion | 153 |
| reflective_metal | 139 |
| transparent_glass | 108 |

### Category counts

| Category | Objects |
|---|---:|
| robot_gripper | 180 |
| plastic_object | 91 |
| metal_part | 90 |
| connector | 84 |
| screw | 80 |
| box | 68 |
| tool | 60 |
| cable | 59 |
| glass_object | 52 |

## SAM ViT-B prompt comparison

| Prompt | Objects | Mean IoU | Median IoU | Boundary F1 | IoU >= 0.90 | IoU >= 0.75 | IoU >= 0.50 | IoU < 0.10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Box | 764 | 0.9052 | 0.9576 | 0.9331 | 0.7474 | 0.9123 | 0.9607 | 0.0039 |
| Point | 764 | 0.8021 | 0.9144 | 0.8124 | 0.5301 | 0.7304 | 0.8652 | 0.0249 |
| Auto | 764 | 0.7925 | 0.9416 | 0.8267 | 0.6008 | 0.7356 | 0.8547 | 0.0628 |

## Interpretation

SAM ViT-B with box prompts is the strongest and most reliable prompt mode on the v2-clean-medium dataset. It reaches 0.9052 mean IoU and has a very low catastrophic failure rate of 0.39%.

Point prompts are usable but less stable, especially on ambiguous, thin, transparent, or partially occluded objects.

Automatic mask generation has a high median IoU but a higher catastrophic failure rate. This shows that automatic full-image mask proposals often segment easy objects well but can miss individual target instances.

The v2-clean-medium dataset is suitable as the main medium-scale benchmark for development. The next scale-up step is to generate a larger final dataset by producing more candidate images and filtering them with the same audit pipeline.

