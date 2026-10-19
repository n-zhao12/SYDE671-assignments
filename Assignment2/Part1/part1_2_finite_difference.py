from pathlib import Path

import numpy as np
from scipy.signal import convolve2d

from part1_utils import DX, DY, normalize_for_display, save_image_grid


def finite_difference(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = convolve2d(image, DX, mode="same", boundary="fill", fillvalue=0)
    gy = convolve2d(image, DY, mode="same", boundary="fill", fillvalue=0)
    return gx, gy


def gradient_magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    return np.sqrt(gx ** 2 + gy ** 2)


def run_finite_difference_section(
    image: np.ndarray,
    output_dir: Path,
    threshold: float | None = None,
) -> dict[str, np.ndarray]:
    gx, gy = finite_difference(image)
    magnitude = gradient_magnitude(gx, gy)
    if threshold is None:
        threshold = float(np.percentile(magnitude, 85))
    edges = magnitude >= threshold
    save_image_grid(output_dir / "part1_2_finite_difference.png", [
        ("Cameraman", image),
        ("Dx", normalize_for_display(gx)),
        ("Dy", normalize_for_display(gy)),
        ("Gradient magnitude", normalize_for_display(magnitude)),
        (f"Binary edges (T={threshold:.1f})", edges),
    ])
    return {"gx": gx, "gy": gy, "magnitude": magnitude, "edges": edges,
            "threshold": np.array(threshold)}