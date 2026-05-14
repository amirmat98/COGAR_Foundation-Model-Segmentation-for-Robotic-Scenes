# Simulated Dataset Sanity Benchmark

## Dataset

- Generator: BlenderProc COGAR-Sim pipeline
- Raw generated images: 25
- Suspicious/dark frames excluded: 4
- Final usable object instances: 360
- Image resolution: 640 x 480
- Categories: plastic_object, connector, screw, robot_gripper, cable, box, metal_part, glass_object, tool
- Challenge groups: reflective_metal, transparent_glass, partial_occlusion, small_parts, dynamic_scene

## SAM ViT-B Results

| Prompt mode | Objects | Mean IoU | Median IoU | Boundary F1 | Mean latency | FPS |
|---|---:|---:|---:|---:|---:|---:|
| Box | 360 | 0.8885 | 0.9422 | 0.9367 | 0.0190 s | 78.95 |
| Point | 360 | 0.7993 | 0.9154 | 0.8330 | 0.0199 s | 75.04 |
| Automatic | 360 | 0.7527 | 0.9289 | 0.8067 | 13.20 s | 0.076 |

## Initial observations

Bounding-box prompts performed best overall. Single-point prompts were faster but less reliable, especially for small or ambiguous objects. Automatic mask generation produced high median IoU but many near-zero failures, showing that it often failed to propose the correct object instance separately.

## Notable failure cases

- Glass objects: very low IoU in several box, point, and automatic-mask cases.
- Cables: frequent low-IoU cases, likely because they are thin and elongated.
- Robot gripper parts: occasional poor masks when partially occluded or visually merged with dark objects.
- Small screws/connectors: point and automatic prompts failed more often than box prompts.
