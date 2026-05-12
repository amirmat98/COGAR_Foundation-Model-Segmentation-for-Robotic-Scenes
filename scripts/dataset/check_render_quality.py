from pathlib import Path

import cv2
import numpy as np


IMAGE_ROOT = Path("data/cogar_sim_500/raw_blenderproc/pilot_v2_ocid_like/coco_data")

image_paths = sorted(
    list(IMAGE_ROOT.rglob("*.png")) + list(IMAGE_ROOT.rglob("*.jpg"))
)

print(f"Found {len(image_paths)} images")

bad = []

for path in image_paths:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        bad.append((path, "unreadable"))
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())

    # Heuristic: very dark or almost flat images are suspicious.
    if mean < 20 or std < 8:
        bad.append((path, f"mean={mean:.2f}, std={std:.2f}"))

print("\nSuspicious frames:")
for path, reason in bad:
    print(path, reason)

print(f"\nBad/suspicious count: {len(bad)}")
