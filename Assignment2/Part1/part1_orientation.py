from pathlib import Path

import cv2
import numpy as np

from part1_2_finite_difference import gradient_magnitude
from part1_utils import save_image


def gradient_orientation_without_arctan2(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Use arctan of the component ratio and correct its quadrant explicitly."""
    angle = np.zeros_like(gx, dtype=np.float64)
    nonzero = (gx != 0) | (gy != 0)
    regular = gx != 0
    vertical = (gx == 0) & (gy != 0)
    angle[regular] = np.arctan(gy[regular] / gx[regular])
    angle[regular & (gx < 0)] += np.pi
    angle[vertical & (gy > 0)] = np.pi / 2
    angle[vertical & (gy < 0)] = 3 * np.pi / 2
    angle[nonzero] %= 2 * np.pi
    return angle


def run_orientation_section(gx: np.ndarray, gy: np.ndarray, output_dir: Path) -> np.ndarray:
    angle = gradient_orientation_without_arctan2(gx, gy)
    magnitude = gradient_magnitude(gx, gy)
    max_magnitude = float(np.max(magnitude))
    value = magnitude / max_magnitude if max_magnitude > 0 else np.zeros_like(magnitude)
    hue_degrees = angle * (180.0 / np.pi)
    hsv = np.stack([hue_degrees, np.ones_like(angle), value], axis=-1).astype(np.float32)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    save_image(output_dir / "part1_bells_gradient_orientation.png", rgb,
               "Gradient orientation (hue) and strength (brightness)")
    return angle