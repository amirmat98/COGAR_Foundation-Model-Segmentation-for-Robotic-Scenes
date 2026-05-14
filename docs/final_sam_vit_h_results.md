# SAM ViT-H Hardware-Limited Evaluation

## Checkpoint verification

The official SAM ViT-H checkpoint was downloaded and verified successfully.

- Checkpoint: `checkpoints/sam_vit_h_4b8939.pth`
- Size: 2,564,550,879 bytes
- PyTorch load test: successful
- Number of tensors: 594

## Full GPU evaluation

Full SAM ViT-H evaluation on the final 500-image dataset could not be completed on the available GPU.

The available GPU has approximately 4 GB VRAM. During ViT-H image encoding, PyTorch raised a CUDA out-of-memory error. Therefore, the full SAM prompt benchmark is reported using SAM ViT-B, which successfully completed box, point, and automatic-mask evaluation over all 4,471 object instances.

## CPU subset proof run

A CPU-only ViT-H box-prompt run was completed successfully on a small 25-object subset.

| Model | Prompt | Device | Objects | Mean IoU | Median IoU | Mean SAM Score |
|---|---|---|---:|---:|---:|---:|
| SAM ViT-H | Box | CPU | 25 | 0.9820 | 0.9862 | 0.9851 |

The subset was balanced by primary challenge, with 5 examples per challenge. However, because all 25 selected instances were `box` objects, this run is treated only as a hardware-feasibility proof, not as a full category-balanced benchmark.

## Interpretation

The ViT-H checkpoint and implementation are valid, but full evaluation is hardware-limited on the available laptop GPU. The CPU subset confirms that ViT-H can run correctly, but CPU inference is too slow for the full 4,471-object benchmark. For this project, SAM ViT-B is used as the complete SAM benchmark model, while SAM ViT-H is documented as a hardware-limited larger variant.

