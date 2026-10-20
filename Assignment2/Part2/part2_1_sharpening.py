from pathlib import Path

import cv2
import numpy as np

from part2_common import gaussian_blur, read_rgb, save_grid, signed_display


def unsharp_kernel(sigma: float = 3.0, amount: float = 1.0) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    size = 2 * int(np.ceil(4 * sigma)) + 1
    gaussian_1d = cv2.getGaussianKernel(size, sigma, cv2.CV_64F)
    kernel = -amount * (gaussian_1d @ gaussian_1d.T)
    kernel[size // 2, size // 2] += 1.0 + amount
    return kernel


def unsharp_mask(image: np.ndarray, sigma: float = 3.0,
                 amount: float = 1.0) -> np.ndarray:
    """Apply one convolution; the symmetric kernel also permits correlation."""
    return cv2.filter2D(image.astype(np.float64), cv2.CV_64F,
                        unsharp_kernel(sigma, amount), borderType=cv2.BORDER_REFLECT101)


def run_sharpening(path: str | Path, name: str, output_dir: Path) -> None:
    image = read_rgb(path).astype(np.float64)
    blurred = gaussian_blur(image, sigma=3.0)
    high_frequency = image - blurred
    panels = [
        ("Original", image),
        ("Gaussian blur", blurred),
        ("High frequencies", signed_display(high_frequency)),
    ]
    for amount in (0.5, 1.0, 1.5):
        filtered = unsharp_mask(image, sigma=3.0, amount=amount)
        sharpened = np.clip(filtered, 0.0, 1.0)
        panels.append((f"Unsharp amount {amount:g}", sharpened))
    save_grid(output_dir / f"{name}_unsharp.png", panels, columns=3)

    blurred_sharp = blurred
    restored = np.clip(unsharp_mask(blurred_sharp, sigma=3.0, amount=1.5), 0.0, 1.0)
    save_grid(output_dir / f"{name}_blur_then_sharpen.png", [
        ("Original", image),
        ("Blurred", blurred_sharp),
        ("Sharpened again", restored),
    ], columns=3)
