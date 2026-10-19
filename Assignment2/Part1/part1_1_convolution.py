import time
from pathlib import Path

import numpy as np
from scipy.signal import convolve2d

from part1_utils import DX, DY, normalize_for_display, read_grayscale, resize_max_side, save_image_grid


def convolve2d_four_loops(image: np.ndarray, kernel: np.ndarray,
                          padding: bool = True) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("image and kernel must both be 2D arrays")

    kh, kw = kernel.shape
    if padding:
        pad_top, pad_left = kh // 2, kw // 2
        pad_bottom, pad_right = kh - 1 - pad_top, kw - 1 - pad_left
        padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode="constant", constant_values=0)
    else:
        padded = image

    flipped = kernel[::-1, ::-1]
    out_h = padded.shape[0] - kh + 1
    out_w = padded.shape[1] - kw + 1
    output = np.zeros((out_h, out_w), dtype=np.float64)
    for y in range(out_h):
        for x in range(out_w):
            total = 0.0
            for ky in range(kh):
                for kx in range(kw):
                    total += padded[y + ky, x + kx] * flipped[ky, kx]
            output[y, x] = total
    return output


def convolve2d_two_loops(image: np.ndarray, kernel: np.ndarray,
                         padding: bool = True) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("image and kernel must both be 2D arrays")

    kh, kw = kernel.shape
    if padding:
        pad_top, pad_left = kh // 2, kw // 2
        pad_bottom, pad_right = kh - 1 - pad_top, kw - 1 - pad_left
        padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode="constant", constant_values=0)
    else:
        padded = image

    flipped = kernel[::-1, ::-1]
    out_h = padded.shape[0] - kh + 1
    out_w = padded.shape[1] - kw + 1
    output = np.zeros((out_h, out_w), dtype=np.float64)
    for y in range(out_h):
        for x in range(out_w):
            patch = padded[y:y + kh, x:x + kw]
            output[y, x] = np.sum(patch * flipped)
    return output


def test_convolutions() -> None:
    image = np.arange(1, 21, dtype=np.float64).reshape(4, 5)
    kernel = np.array([[1, 2, 0], [-1, 0, 1]], dtype=np.float64)
    expected = convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0)
    result_four = convolve2d_four_loops(image, kernel)
    result_two = convolve2d_two_loops(image, kernel)
    assert np.allclose(result_four, expected)
    assert np.allclose(result_two, expected)
    print("Convolution correctness checks passed against scipy.signal.convolve2d.")


def run_selfie_section(selfie_path: str, output_dir: Path) -> None:
    image = resize_max_side(read_grayscale(selfie_path), max_side=512)
    box = np.ones((9, 9), dtype=np.float64) / 81.0
    blurred = convolve2d_two_loops(image, box)
    derivative_x = convolve2d_two_loops(image, DX)
    derivative_y = convolve2d_two_loops(image, DY)
    save_image_grid(output_dir / "part1_1_selfie.png", [
        ("Original grayscale", image),
        ("9x9 box filter", blurred),
        ("Dx response", normalize_for_display(derivative_x)),
        ("Dy response", normalize_for_display(derivative_y)),
    ])


def benchmark_convolutions(selfie_path: str, output_dir: Path) -> None:
    image = resize_max_side(read_grayscale(selfie_path), max_side=128)
    kernel = np.ones((9, 9), dtype=np.float64) / 81.0
    implementations = [
        ("Four loops", convolve2d_four_loops),
        ("Two loops", convolve2d_two_loops),
        ("SciPy", lambda im, k: convolve2d(im, k, mode="same", boundary="fill", fillvalue=0)),
    ]
    print(f"Part 1.1 timing size: {image.shape[1]}x{image.shape[0]}")
    for name, function in implementations:
        print(f"Timing {name}...", flush=True)
        start = time.perf_counter()
        function(image, kernel)
        print(f"{name}: {time.perf_counter() - start:.4f} seconds")
    run_selfie_section(selfie_path, output_dir)