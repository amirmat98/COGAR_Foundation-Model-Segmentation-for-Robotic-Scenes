"""SAM model loading and inference helpers.

The heavy dependencies are imported inside functions so pure unit tests can
run without importing PyTorch or Segment Anything unless SAM is actually used.
"""

from pathlib import Path
from typing import Any, Literal

import numpy as np

DeviceMode = Literal["auto", "cpu", "cuda"]


def select_device(requested_device: DeviceMode, allow_cpu_fallback: bool) -> str:
    """
    Select CPU or CUDA and verify CUDA with a real tensor operation.

    ``torch.cuda.is_available()`` can be true even when the installed CUDA build
    is incompatible with the local GPU, so this performs a small tensor test.
    """
    import torch

    if requested_device == "cpu":
        return "cpu"

    if not torch.cuda.is_available():
        if requested_device == "cuda" and not allow_cpu_fallback:
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

        print("CUDA is not available. Falling back to CPU.")
        return "cpu"

    try:
        test_tensor = torch.empty(1, device="cuda")
        test_tensor += 1
        torch.cuda.synchronize()
        return "cuda"

    except Exception as exc:
        message = (
            "CUDA is visible, but a real CUDA tensor test failed.\n"
            "This usually means your PyTorch CUDA build is not compatible with your GPU.\n"
            f"Original CUDA error: {exc}"
        )

        if requested_device == "cuda" and not allow_cpu_fallback:
            raise RuntimeError(message) from exc

        print(message)
        print("Falling back to CPU.")
        return "cpu"


def load_sam_predictor(checkpoint_path: str | Path, model_type: str, device: str) -> Any:
    """Load a SAM predictor for a checkpoint and model type."""
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device)
    sam.eval()

    return SamPredictor(sam)


def run_sam_for_box(predictor: Any, box_xyxy: np.ndarray) -> tuple[np.ndarray, float]:
    """Run prediction for the current predictor image and one XYXY box."""
    import torch

    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            box=box_xyxy,
            multimask_output=False,
        )

    return masks[0], float(scores[0])


def run_sam_for_point(
    predictor: Any,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run prediction for the current predictor image and one point prompt.

    Args:
        predictor:
            A SAM predictor with an image already set by predictor.set_image(...).

        point_coords:
            NumPy array with shape (N, 2), containing point prompts in XY pixel format.

        point_labels:
            NumPy array with shape (N,), where 1 means foreground and 0 means background.

    Returns:
        A tuple containing:
            - the selected SAM binary mask
            - the SAM predicted mask-quality score
    """
    import torch

    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )

    return masks[0], float(scores[0])

def run_sam_box_prompt(
    image_rgb: np.ndarray,
    box_xyxy: np.ndarray,
    checkpoint_path: str | Path,
    model_type: str,
    device: str,
) -> tuple[np.ndarray, float]:
    """Load SAM, set one RGB image, and predict one box-prompt mask."""
    predictor = load_sam_predictor(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        device=device,
    )
    predictor.set_image(image_rgb)
    return run_sam_for_box(predictor, box_xyxy)
