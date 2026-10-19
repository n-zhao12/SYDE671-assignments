from pathlib import Path

import cv2
import numpy as np
from scipy.signal import convolve2d

from part1_2_finite_difference import finite_difference, gradient_magnitude
from part1_utils import DX, DY, normalize_for_display, save_image_grid


def gaussian_kernel(size: int = 9, sigma: float = 1.5) -> np.ndarray:
    if size <= 0 or size % 2 == 0:
        raise ValueError("Gaussian kernel size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    one_dimensional = cv2.getGaussianKernel(size, sigma, cv2.CV_64F)
    kernel = one_dimensional @ one_dimensional.T
    return kernel / np.sum(kernel)


def smooth_then_differentiate(
    image: np.ndarray, gaussian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retain the zero-extended blur's halo until after differentiation."""
    full_blur = convolve2d(image, gaussian, mode="full", boundary="fill", fillvalue=0)
    full_gx, full_gy = finite_difference(full_blur)
    offset_y, offset_x = gaussian.shape[0] // 2, gaussian.shape[1] // 2
    crop = (slice(offset_y, offset_y + image.shape[0]),
            slice(offset_x, offset_x + image.shape[1]))
    return full_blur[crop], full_gx[crop], full_gy[crop]


def run_dog_section(
    image: np.ndarray,
    output_dir: Path,
    gaussian_size: int = 9,
    sigma: float = 1.5,
    threshold: float | None = None,
) -> dict[str, np.ndarray]:
    gaussian = gaussian_kernel(gaussian_size, sigma)
    smoothed, gx_smooth, gy_smooth = smooth_then_differentiate(image, gaussian)
    magnitude_smooth = gradient_magnitude(gx_smooth, gy_smooth)
    dog_x = convolve2d(gaussian, DX, mode="full")
    dog_y = convolve2d(gaussian, DY, mode="full")
    gx_dog = convolve2d(image, dog_x, mode="same", boundary="fill", fillvalue=0)
    gy_dog = convolve2d(image, dog_y, mode="same", boundary="fill", fillvalue=0)
    magnitude_dog = gradient_magnitude(gx_dog, gy_dog)
    np.testing.assert_allclose(gx_smooth, gx_dog, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(gy_smooth, gy_dog, atol=1e-10, rtol=1e-10)
    if threshold is None:
        threshold = float(np.percentile(magnitude_smooth, 85))
    edges_smooth = magnitude_smooth >= threshold
    edges_dog = magnitude_dog >= threshold

    save_image_grid(output_dir / "part1_3_gaussian_dog_kernels.png", [
        ("Gaussian kernel", normalize_for_display(gaussian)),
        ("DoG x kernel", normalize_for_display(dog_x)),
        ("DoG y kernel", normalize_for_display(dog_y)),
    ])
    save_image_grid(output_dir / "part1_3_dog_results.png", [
        ("Original", image),
        ("Gaussian smoothed", smoothed),
        ("Smooth then Dx", normalize_for_display(gx_smooth)),
        ("Smooth then Dy", normalize_for_display(gy_smooth)),
        ("Smooth gradient magnitude", normalize_for_display(magnitude_smooth)),
        (f"Smoothed binary edges (T={threshold:.1f})", edges_smooth),
    ])
    save_image_grid(output_dir / "part1_3_dog_direct_comparison.png", [
        ("Smooth then Dx", gx_smooth),
        ("Direct DoG x", gx_dog),
        ("Smooth then Dy", gy_smooth),
        ("Direct DoG y", gy_dog),
        ("Smooth-then-differentiate magnitude", magnitude_smooth),
        ("Direct DoG magnitude", magnitude_dog),
        (f"Direct DoG binary edges (T={threshold:.1f})", edges_dog),
    ])
    print("Max x-response difference:", float(np.max(np.abs(gx_smooth - gx_dog))))
    print("Max y-response difference:", float(np.max(np.abs(gy_smooth - gy_dog))))
    return {
        "gaussian": gaussian, "smoothed": smoothed,
        "dog_x": dog_x, "dog_y": dog_y,
        "gx_smooth": gx_smooth, "gy_smooth": gy_smooth,
        "gx_dog": gx_dog, "gy_dog": gy_dog,
        "magnitude_smooth": magnitude_smooth, "magnitude_dog": magnitude_dog,
        "edges_smooth": edges_smooth, "edges_dog": edges_dog,
    }