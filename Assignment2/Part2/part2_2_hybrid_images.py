from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from part2_common import fit_to_shape, gaussian_blur, gray, read_rgb, save_grid, signed_display


DEREK_EYE_POINTS = np.array([[225.0, 256.0], [327.0, 244.0]], dtype=np.float32)
NUTMEG_EYE_POINTS = np.array([[332.0, 153.0], [410.0, 196.0]], dtype=np.float32)


def select_two_points(image: np.ndarray, title: str) -> np.ndarray:
    print(f"{title}: click two corresponding landmarks in the same order, then close the window.")
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.imshow(image)
    axis.set_title(title)
    axis.axis("off")
    points = plt.ginput(2, timeout=0)
    plt.close(figure)
    if len(points) != 2:
        raise RuntimeError("Select exactly two corresponding landmarks in each image.")
    points = np.asarray(points, dtype=np.float32)
    print(f"{title} landmarks (x, y): {np.rint(points).astype(int).tolist()}")
    return points


def align_pair(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_points: np.ndarray,
    moving_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference_vector = reference_points[1] - reference_points[0]
    moving_vector = moving_points[1] - moving_points[0]
    reference_length = float(np.linalg.norm(reference_vector))
    moving_length = float(np.linalg.norm(moving_vector))
    if reference_length == 0.0 or moving_length == 0.0:
        raise ValueError("Alignment points must be distinct.")

    scale = reference_length / moving_length
    angle = np.arctan2(reference_vector[1], reference_vector[0]) - np.arctan2(
        moving_vector[1], moving_vector[0]
    )
    cosine, sine = np.cos(angle), np.sin(angle)
    linear = scale * np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    translation = reference_points[0] - linear @ moving_points[0]
    transform = np.column_stack((linear, translation)).astype(np.float32)
    height, width = reference.shape[:2]
    warped = cv2.warpAffine(
        moving,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.5, 0.5, 0.5),
    )
    support = cv2.warpAffine(
        np.ones(moving.shape[:2], dtype=np.float32),
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    support = cv2.GaussianBlur(support, (0, 0), sigmaX=8.0, sigmaY=8.0)
    return warped.astype(np.float32), np.clip(support, 0.0, 1.0).astype(np.float32)


def fourier_display(image: np.ndarray) -> np.ndarray:
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(image))))
    low, high = np.percentile(spectrum, (1.0, 99.5))
    if high <= low:
        return np.zeros_like(spectrum, dtype=np.float32)
    return np.clip((spectrum - low) / (high - low), 0.0, 1.0).astype(np.float32)


def run_hybrid(
    path_a: str | Path,
    path_b: str | Path,
    name: str,
    output_dir: Path,
    low_sigma: float,
    high_sigma: float,
    align: bool,
    full_analysis: bool = False,
    landmark_points: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    image_a = read_rgb(path_a)
    original_b = read_rgb(path_b)
    image_b = original_b
    if align:
        if landmark_points is None:
            landmark_points = (
                select_two_points(image_a, f"{name} image A"),
                select_two_points(image_b, f"{name} image B"),
            )
        image_b, highpass_mask = align_pair(image_a, image_b, *landmark_points)
    else:
        image_b = fit_to_shape(image_b, image_a.shape[:2])
        highpass_mask = np.ones(image_a.shape[:2], dtype=np.float32)

    gray_a, gray_b = gray(image_a), gray(image_b)
    low_gray = gaussian_blur(gray_a, low_sigma)
    high_gray = (gray_b - gaussian_blur(gray_b, high_sigma)) * highpass_mask
    hybrid_gray = np.clip(low_gray + high_gray, 0.0, 1.0)
    low_color = gaussian_blur(image_a, low_sigma)
    high_color = (image_b - gaussian_blur(image_b, high_sigma)) * highpass_mask[..., None]
    hybrid_rgb = np.repeat(hybrid_gray[..., None], 3, axis=2)
    if full_analysis:
        color_variants = [
            ("Color high-frequency component", np.clip(low_gray[..., None] + high_color, 0.0, 1.0)),
            ("Color low-frequency component", np.clip(low_color + high_gray[..., None], 0.0, 1.0)),
            ("Both components in color", np.clip(low_color + high_color, 0.0, 1.0)),
        ]
        save_grid(output_dir / f"{name}_01_original_and_aligned.png", [
            ("Original A", image_a),
            ("Original B", original_b),
            ("Aligned B used for filtering", image_b),
        ], columns=3)
        save_grid(output_dir / f"{name}_02_fourier_transforms.png", [
            ("Input A spectrum", fourier_display(gray_a)),
            ("Aligned B spectrum", fourier_display(gray_b)),
            (f"Low-pass spectrum (sigma={low_sigma:g})", fourier_display(low_gray)),
            (f"High-pass spectrum (sigma={high_sigma:g})", fourier_display(high_gray)),
            ("Hybrid spectrum", fourier_display(hybrid_gray)),
        ], columns=3)
        save_grid(output_dir / f"{name}_03_filtered_results.png", [
            (f"Low-pass A (sigma={low_sigma:g})", low_gray),
            (f"High-pass B (sigma={high_sigma:g})", signed_display(high_gray)),
        ], columns=2)

        cutoff_panels = []
        for candidate_sigma in sorted({max(0.5, low_sigma * factor) for factor in (0.5, 1.0, 1.5)}):
            candidate_low = gaussian_blur(gray_a, candidate_sigma)
            candidate_hybrid = np.clip(candidate_low + high_gray, 0.0, 1.0)
            selected = " selected" if np.isclose(candidate_sigma, low_sigma) else ""
            cutoff_panels.append((
                f"Low sigma={candidate_sigma:g}{selected} (high fixed={high_sigma:g})",
                np.repeat(candidate_hybrid[..., None], 3, axis=2),
            ))
        for candidate_sigma in sorted({max(0.5, high_sigma * factor) for factor in (0.5, 1.0, 1.5)}):
            candidate_high = (gray_b - gaussian_blur(gray_b, candidate_sigma)) * highpass_mask
            candidate_hybrid = np.clip(low_gray + candidate_high, 0.0, 1.0)
            selected = " selected" if np.isclose(candidate_sigma, high_sigma) else ""
            cutoff_panels.append((
                f"High sigma={candidate_sigma:g}{selected} (low fixed={low_sigma:g})",
                np.repeat(candidate_hybrid[..., None], 3, axis=2),
            ))
        save_grid(output_dir / f"{name}_04_cutoff_choice.png", cutoff_panels, columns=3)
        save_grid(output_dir / f"{name}_05_final_hybrid.png", [
            (f"Final grayscale hybrid (low sigma={low_sigma:g}, high sigma={high_sigma:g})", hybrid_rgb),
        ], columns=1)
        save_grid(output_dir / f"{name}_color_variants.png", [
            ("Input A", image_a),
            ("Aligned B", image_b),
            ("Grayscale hybrid", hybrid_rgb),
            *color_variants,
        ], columns=3)
    else:
        save_grid(output_dir / f"{name}_hybrids.png", [
            ("Original A", image_a),
            ("Original B", original_b),
            ("Final hybrid", hybrid_rgb),
        ], columns=3)