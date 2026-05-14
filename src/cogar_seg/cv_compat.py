"""Small OpenCV compatibility layer.

The project depends on OpenCV for normal runs. Some lightweight CI/test
environments do not provide ``cv2``; in that case this module supplies the small
subset needed by tests and validation utilities using Pillow/SciPy.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised when OpenCV is installed.
    import cv2 as cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback is covered indirectly.
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    from PIL import Image
    from scipy import ndimage

    class _CV2Fallback:
        IMREAD_COLOR = 1
        IMREAD_GRAYSCALE = 0
        IMREAD_UNCHANGED = -1
        COLOR_BGR2GRAY = 6
        COLOR_BGR2RGB = 4
        COLOR_RGB2BGR = 4
        DIST_L2 = 2
        MORPH_RECT = 0

        @staticmethod
        def imread(path: str, flags: int = 1):
            try:
                image = Image.open(path)
            except (FileNotFoundError, OSError):
                return None

            if flags == _CV2Fallback.IMREAD_GRAYSCALE:
                return np.asarray(image.convert("L"), dtype=np.uint8)
            if flags == _CV2Fallback.IMREAD_UNCHANGED:
                array = np.asarray(image)
                if array.ndim == 3 and array.shape[2] >= 3:
                    array = array.copy()
                    array[..., :3] = array[..., 2::-1]
                return array

            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            return array[..., ::-1].copy()

        @staticmethod
        def imwrite(path: str, image) -> bool:
            try:
                array = np.asarray(image)
                if array.dtype == bool:
                    array = array.astype(np.uint8) * 255
                if array.ndim == 3 and array.shape[2] >= 3:
                    array = array.copy()
                    array[..., :3] = array[..., 2::-1]
                Image.fromarray(array.astype(np.uint8)).save(Path(path))
                return True
            except OSError:
                return False

        @staticmethod
        def cvtColor(image, code: int):
            array = np.asarray(image)
            if code in {_CV2Fallback.COLOR_BGR2RGB, _CV2Fallback.COLOR_RGB2BGR}:
                return array[..., ::-1].copy()
            if code == _CV2Fallback.COLOR_BGR2GRAY:
                b = array[..., 0].astype(float)
                g = array[..., 1].astype(float)
                r = array[..., 2].astype(float)
                return np.clip(0.114 * b + 0.587 * g + 0.299 * r, 0, 255).astype(np.uint8)
            raise ValueError(f"Unsupported cvtColor code in fallback: {code}")

        @staticmethod
        def getStructuringElement(shape: int, ksize: tuple[int, int]):
            del shape
            width, height = ksize
            return np.ones((height, width), dtype=bool)

        @staticmethod
        def erode(image, kernel, iterations: int = 1):
            result = np.asarray(image) > 0
            for _ in range(iterations):
                result = ndimage.binary_erosion(result, structure=kernel)
            return result.astype(np.asarray(image).dtype)

        @staticmethod
        def dilate(image, kernel, iterations: int = 1):
            result = np.asarray(image) > 0
            for _ in range(iterations):
                result = ndimage.binary_dilation(result, structure=kernel)
            return result.astype(np.asarray(image).dtype)

        @staticmethod
        def distanceTransform(image, distance_type: int, mask_size: int):
            del distance_type, mask_size
            return ndimage.distance_transform_edt(np.asarray(image) > 0).astype(np.float32)

        @staticmethod
        def rectangle(image, pt1, pt2, color, thickness: int):
            x1, y1 = pt1
            x2, y2 = pt2
            arr = image
            if thickness < 0:
                arr[y1:y2 + 1, x1:x2 + 1] = color
                return arr
            for offset in range(thickness):
                arr[y1 + offset, x1:x2 + 1] = color
                arr[y2 - offset, x1:x2 + 1] = color
                arr[y1:y2 + 1, x1 + offset] = color
                arr[y1:y2 + 1, x2 - offset] = color
            return arr

        @staticmethod
        def circle(image, center, radius: int, color, thickness: int):
            cx, cy = center
            arr = image
            yy, xx = np.ogrid[: arr.shape[0], : arr.shape[1]]
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            if thickness < 0:
                mask = dist <= radius
            else:
                mask = np.logical_and(dist <= radius, dist >= max(0, radius - thickness))
            arr[mask] = color
            return arr

    cv2 = SimpleNamespace(
        IMREAD_COLOR=_CV2Fallback.IMREAD_COLOR,
        IMREAD_GRAYSCALE=_CV2Fallback.IMREAD_GRAYSCALE,
        IMREAD_UNCHANGED=_CV2Fallback.IMREAD_UNCHANGED,
        COLOR_BGR2GRAY=_CV2Fallback.COLOR_BGR2GRAY,
        COLOR_BGR2RGB=_CV2Fallback.COLOR_BGR2RGB,
        COLOR_RGB2BGR=_CV2Fallback.COLOR_RGB2BGR,
        DIST_L2=_CV2Fallback.DIST_L2,
        MORPH_RECT=_CV2Fallback.MORPH_RECT,
        imread=_CV2Fallback.imread,
        imwrite=_CV2Fallback.imwrite,
        cvtColor=_CV2Fallback.cvtColor,
        getStructuringElement=_CV2Fallback.getStructuringElement,
        erode=_CV2Fallback.erode,
        dilate=_CV2Fallback.dilate,
        distanceTransform=_CV2Fallback.distanceTransform,
        rectangle=_CV2Fallback.rectangle,
        circle=_CV2Fallback.circle,
    )
