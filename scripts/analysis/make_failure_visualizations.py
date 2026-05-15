import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


RUNS = {
    "sam_vit_b_box": "outputs/tables/final_cross_model_analysis/sam_vitb_box_worst_25.csv",
    "mobilesam_box": "outputs/tables/final_cross_model_analysis/mobilesam_box_worst_25.csv",
    "fastsam_s_box": "outputs/tables/final_cross_model_analysis/fastsams_box_worst_25.csv",
}


def read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 0


def find_pred_mask_path(row: pd.Series) -> str:
    for col in ["mask_output_path", "sam_mask_path", "pred_mask_path"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    raise KeyError(f"No prediction mask path found. Columns: {list(row.index)}")


def resize_bool(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if mask.shape[:2] == (h, w):
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0


def overlay_masks(image_rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    gt = resize_bool(gt, (h, w))
    pred = resize_bool(pred, (h, w))

    out = image_rgb.copy().astype(np.float32)

    gt_only = gt & ~pred
    pred_only = pred & ~gt
    overlap = gt & pred

    out[gt_only] = 0.55 * out[gt_only] + 0.45 * np.array([0, 255, 0])
    out[pred_only] = 0.55 * out[pred_only] + 0.45 * np.array([255, 0, 0])
    out[overlap] = 0.55 * out[overlap] + 0.45 * np.array([255, 255, 0])

    return np.clip(out, 0, 255).astype(np.uint8)


def mask_to_rgb(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask.astype(bool)] = np.array(color, dtype=np.uint8)
    return out


def add_title(img: np.ndarray, title: str) -> np.ndarray:
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    canvas = cv2.copyMakeBorder(img_bgr, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(
        canvas,
        title[:100],
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def make_panel(row: pd.Series, out_path: Path, max_width: int = 320) -> None:
    image = read_rgb(str(row["image_path"]))
    gt = read_mask(str(row["gt_mask_path"]))
    pred = read_mask(find_pred_mask_path(row))

    h, w = image.shape[:2]
    scale = max_width / w
    new_h = int(round(h * scale))

    image_small = cv2.resize(image, (max_width, new_h), interpolation=cv2.INTER_AREA)
    gt_small = resize_bool(gt, (new_h, max_width))
    pred_small = resize_bool(pred, (new_h, max_width))

    overlay = overlay_masks(image_small, gt_small, pred_small)

    title = (
        f"IoU={float(row['iou']):.3f} | "
        f"{row.get('category_name', 'unknown')} | "
        f"{row.get('challenge_primary', 'unknown')} | "
        f"{row.get('file_name', '')}"
    )

    panels = [
        add_title(image_small, "RGB"),
        add_title(mask_to_rgb(gt_small, (0, 255, 0)), "GT mask"),
        add_title(mask_to_rgb(pred_small, (255, 0, 0)), "Pred mask"),
        add_title(overlay, title),
    ]

    combined = np.concatenate(panels, axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/figures/failure_modes")
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for run_name, csv_path in RUNS.items():
        p = Path(csv_path)
        if not p.exists():
            print(f"[missing] {p}")
            continue

        df = pd.read_csv(p).sort_values("iou").head(args.top_k).reset_index(drop=True)
        run_dir = out_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        for i, row in df.iterrows():
            category = str(row.get("category_name", "unknown"))
            challenge = str(row.get("challenge_primary", "unknown"))
            iou = float(row["iou"])

            safe_category = category.replace("/", "_")
            safe_challenge = challenge.replace("/", "_")
            out_path = run_dir / f"worst_{i+1:02d}_iou_{iou:.3f}_{safe_category}_{safe_challenge}.png"

            make_panel(row, out_path)

            summary_rows.append({
                "run": run_name,
                "rank": i + 1,
                "iou": iou,
                "category_name": category,
                "challenge_primary": challenge,
                "file_name": row.get("file_name"),
                "object_id": row.get("object_id"),
                "figure_path": str(out_path),
            })

        print(f"[OK] Wrote {len(df)} panels for {run_name}: {run_dir}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_root / "failure_mode_visualization_index.csv", index=False)

    if not summary.empty:
        print("\nFailure summary by category:")
        print(pd.crosstab(summary["category_name"], summary["run"]))

        print("\nFailure summary by challenge:")
        print(pd.crosstab(summary["challenge_primary"], summary["run"]))

    print(f"\n[OK] Output root: {out_root}")


if __name__ == "__main__":
    main()
