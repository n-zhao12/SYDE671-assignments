from pathlib import Path

import cv2
import numpy as np

from part1_2_finite_difference import gradient_magnitude
from part1_utils import save_image


def gradient_orientation_without_arctan2(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Evaluate the inverse-tangent series after half-angle range reduction."""
    abs_x, abs_y = np.abs(gx), np.abs(gy)
    largest = np.maximum(abs_x, abs_y)
    ratio = np.divide(np.minimum(abs_x, abs_y), largest,
                      out=np.zeros_like(largest, dtype=np.float64), where=largest > 0)
    reduced = ratio / (1.0 + np.sqrt(1.0 + ratio ** 2))
    power = reduced.copy()
    angle = np.zeros_like(reduced)
    for index in range(16):
        angle += power / (2 * index + 1)
        power *= -(reduced ** 2)
    angle *= 2.0
    angle = np.where(abs_y > abs_x, np.pi / 2 - angle, angle)
    angle = np.where(gx < 0, np.pi - angle, angle)
    angle = np.where(gy < 0, 2 * np.pi - angle, angle)
    return angle % (2 * np.pi)


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