"""Assignment 2 Parts 2.3 and 2.4: image stacks and multiresolution blends."""

from pathlib import Path

import cv2
import numpy as np

from part2_common import OUTPUT_DIR, ROOT, fit_to_shape, gaussian_blur, gray, read_rgb, save_grid, signed_display


LEVELS = 5
STACK_SIGMA = 2.0
MASK_SIGMA = 1.0
IMAGE_DIR = ROOT / "images"
BLEND_OUTPUT = OUTPUT_DIR / "2_4_blending"
STACK_OUTPUT = OUTPUT_DIR / "2_3_stacks"

CUSTOM_PAIRS = (
	("tennis_volleyball", "blendPair1_tennis_ball.jpg", "blendPair1_volleyball.jpg", 11),
)


def gaussian_stack(image: np.ndarray, levels: int = LEVELS,
				   sigma: float = STACK_SIGMA) -> list[np.ndarray]:
	"""Double the effective blur scale at each level without subsampling."""
	if levels < 1:
		raise ValueError("A stack must contain at least one level")
	if sigma <= 0:
		raise ValueError("sigma must be positive")
	stack = [image.astype(np.float32, copy=True)]
	previous_sigma = 0.0
	for level in range(1, levels):
		target_sigma = sigma * 2 ** (level - 1)
		incremental_sigma = float(np.sqrt(target_sigma ** 2 - previous_sigma ** 2))
		stack.append(gaussian_blur(stack[-1], incremental_sigma))
		previous_sigma = target_sigma
	return stack


def laplacian_stack(gaussian: list[np.ndarray]) -> list[np.ndarray]:
	"""Difference neighboring Gaussian levels; retain the coarsest residual."""
	if not gaussian:
		raise ValueError("Cannot form a Laplacian stack from an empty stack")
	bands = [gaussian[index] - gaussian[index + 1]
			 for index in range(len(gaussian) - 1)]
	bands.append(gaussian[-1])
	return bands


def gaussian_mask_stack(mask: np.ndarray) -> list[np.ndarray]:
	"""Blur the mask before stacking so even the finest band gets a soft seam."""
	return gaussian_stack(gaussian_blur(mask.astype(np.float32), MASK_SIGMA),
						  LEVELS, STACK_SIGMA)


def laplacian_blend(
	laplacian_a: list[np.ndarray],
	laplacian_b: list[np.ndarray],
	mask_stack: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
	"""Blend corresponding Laplacian bands and sum them to reconstruct the image."""
	if not (len(laplacian_a) == len(laplacian_b) == len(mask_stack)):
		raise ValueError("Both image stacks and the mask stack must have equal levels")
	blended_bands = []
	for band_a, band_b, mask in zip(laplacian_a, laplacian_b, mask_stack):
		weight = mask[..., None] if band_a.ndim == 3 else mask
		blended_bands.append(weight * band_a + (1.0 - weight) * band_b)
	return np.sum(np.stack(blended_bands, axis=0), axis=0), blended_bands


def step_mask(height: int, width: int, orientation: str = "vertical") -> np.ndarray:
	mask = np.zeros((height, width), dtype=np.float32)
	if orientation == "vertical":
		mask[:, :width // 2] = 1.0
	elif orientation == "horizontal":
		mask[:height // 2, :] = 1.0
	else:
		raise ValueError("orientation must be 'vertical' or 'horizontal'")
	return mask


def galaxy_silhouette_mask(image: np.ndarray) -> np.ndarray:
	"""Segment the main bright galaxy mass into a filled binary silhouette."""
	height, width = image.shape[:2]
	gray_image = np.clip(gray(image), 0.0, 1.0)
	gray_u8 = np.round(gray_image * 255).astype(np.uint8)
	blur_sigma = max(1.0, min(height, width) * 0.012)
	blurred = cv2.GaussianBlur(gray_u8, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
	_, thresholded = cv2.threshold(
		blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
	)

	kernel_size = max(5, int(round(min(height, width) * 0.055)))
	if kernel_size % 2 == 0:
		kernel_size += 1
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
	closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=2)
	component_count, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
	if component_count <= 1:
		raise ValueError("Could not segment a foreground galaxy component")
	largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
	component = np.where(labels == largest_label, 255, 0).astype(np.uint8)
	contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		raise ValueError("Galaxy segmentation did not produce an external contour")
	contour = max(contours, key=cv2.contourArea)
	epsilon = max(1.0, 0.0015 * cv2.arcLength(contour, True))
	contour = cv2.approxPolyDP(contour, epsilon, True)
	mask = np.zeros((height, width), dtype=np.uint8)
	cv2.fillPoly(mask, [contour], 1)
	smoothing_sigma = max(1.0, min(height, width) * 0.004)
	smoothed = cv2.GaussianBlur(mask, (0, 0), smoothing_sigma)
	return (smoothed >= 0.5).astype(np.float32)


def _image_laplacian(image: np.ndarray) -> list[np.ndarray]:
	return laplacian_stack(gaussian_stack(image))


def _save_stack_visuals(label: str, image: np.ndarray, output_dir: Path) -> list[np.ndarray]:
	gaussian = gaussian_stack(image)
	laplacian = laplacian_stack(gaussian)
	save_grid(output_dir / f"{label}_gaussian_stack.png", [
		(f"G{level}", item) for level, item in enumerate(gaussian)
	], columns=LEVELS)
	save_grid(output_dir / f"{label}_laplacian_stack.png", [
		(f"L{level}", signed_display(item) if level < LEVELS - 1 else np.clip(item, 0, 1))
		for level, item in enumerate(laplacian)
	], columns=LEVELS)
	return laplacian


def blend_process_panels(
	laplacian_a: list[np.ndarray], laplacian_b: list[np.ndarray],
	mask_stack: list[np.ndarray], source_names: tuple[str, str] = ("A", "B"),
) -> list[tuple[str, np.ndarray]]:
	"""Group every weighted band into three rows, followed by reconstruction."""
	levels = len(laplacian_a)
	if levels < 3 or not (levels == len(laplacian_b) == len(mask_stack)):
		raise ValueError("Process figures require matching stacks with at least three levels")
	contributions_a = [mask_level[..., None] * band
					   for mask_level, band in zip(mask_stack, laplacian_a)]
	contributions_b = [(1.0 - mask_level[..., None]) * band
					   for mask_level, band in zip(mask_stack, laplacian_b)]
	split = (levels - 1) // 2
	panels = []
	for start, stop, band_name, letters in (
		(0, split, "High", ("a", "b", "c")),
		(split, levels - 1, "Medium", ("d", "e", "f")),
		(levels - 1, levels, "Low", ("g", "h", "i")),
	):
		contribution_a = np.sum(contributions_a[start:stop], axis=0)
		contribution_b = np.sum(contributions_b[start:stop], axis=0)
		contribution_sum = contribution_a + contribution_b
		if stop < levels:
			scale = float(np.percentile(np.abs(np.stack([
				contribution_a, contribution_b, contribution_sum,
			])), 99.5))
			scale = max(scale, float(np.finfo(np.float32).eps))
			contribution_a = np.clip(0.5 + contribution_a / (2.0 * scale), 0, 1)
			contribution_b = np.clip(0.5 + contribution_b / (2.0 * scale), 0, 1)
			contribution_sum = np.clip(0.5 + contribution_sum / (2.0 * scale), 0, 1)
		else:
			contribution_a = np.clip(contribution_a, 0.0, 1.0)
			contribution_b = np.clip(contribution_b, 0.0, 1.0)
			contribution_sum = np.clip(contribution_sum, 0.0, 1.0)
		panels.extend([
			(f"({letters[0]}) {band_name}: masked {source_names[0]}", contribution_a),
			(f"({letters[1]}) {band_name}: masked {source_names[1]}", contribution_b),
			(f"({letters[2]}) {band_name}: combined", contribution_sum),
		])
	reconstructed_a = np.sum(contributions_a, axis=0)
	reconstructed_b = np.sum(contributions_b, axis=0)
	panels.extend([
		(f"(j) Reconstructed masked {source_names[0]}", np.clip(reconstructed_a, 0, 1)),
		(f"(k) Reconstructed masked {source_names[1]}", np.clip(reconstructed_b, 0, 1)),
		("(l) Final multiresolution blend", np.clip(reconstructed_a + reconstructed_b, 0, 1)),
	])
	return panels


def run_figure_342() -> None:
	"""Recreate Figure 3.42's high-, medium-, low-band Oraple contributions."""
	apple = read_rgb(IMAGE_DIR / "burt_apple.png")
	orange = read_rgb(IMAGE_DIR / "burt_orange.png")
	apple, orange = fit_to_shape(apple, (min(apple.shape[0], orange.shape[0]),
										min(apple.shape[1], orange.shape[1]))), fit_to_shape(
		orange, (min(apple.shape[0], orange.shape[0]), min(apple.shape[1], orange.shape[1]))
	)
	apple_laplacian = _save_stack_visuals("apple", apple, STACK_OUTPUT)
	orange_laplacian = _save_stack_visuals("orange", orange, STACK_OUTPUT)
	height, width = apple.shape[:2]
	hard_mask = step_mask(height, width, "vertical")
	mask_stack = gaussian_mask_stack(hard_mask)
	panels = blend_process_panels(apple_laplacian, orange_laplacian, mask_stack,
								  source_names=("Apple", "Orange"))
	save_grid(STACK_OUTPUT / "oraple_figure_3_42.png", panels, columns=3)


def run_blend_pair(image_a: np.ndarray, image_b: np.ndarray, label: str,
				   mask: np.ndarray, favorite: bool = False,
				   mask_title: str = "Irregular mask") -> np.ndarray:
	image_a, image_b = fit_to_shape(image_a, image_a.shape[:2]), fit_to_shape(
		image_b, image_a.shape[:2]
	)
	mask_stack = gaussian_mask_stack(mask)
	laplacian_a = _image_laplacian(image_a)
	laplacian_b = _image_laplacian(image_b)
	result, _ = laplacian_blend(laplacian_a, laplacian_b, mask_stack)
	result = np.clip(result, 0.0, 1.0)
	soft_mask = mask_stack[0]
	mask_visual = np.repeat(mask_stack[0][..., None], 3, axis=2)

	save_grid(BLEND_OUTPUT / f"{label}_overview.png", [
		("Original A", image_a),
		("Original B", image_b),
		(mask_title, mask_visual),
		("A × soft mask", image_a * soft_mask[..., None]),
		("B × inverse soft mask", image_b * (1.0 - soft_mask[..., None])),
		("Multiresolution blend", result),
	], columns=3)
	save_grid(BLEND_OUTPUT / f"{label}_color_vs_grayscale.png", [
		("Color blend", result),
		("Grayscale blend", np.repeat(
			np.clip(laplacian_blend(_image_laplacian(gray(image_a)),
									_image_laplacian(gray(image_b)), mask_stack)[0], 0, 1)[..., None],
			3,
			axis=2,
		)),
	], columns=2)

	if favorite:
		process_panels = blend_process_panels(laplacian_a, laplacian_b, mask_stack)
		save_grid(BLEND_OUTPUT / f"{label}_laplacian_process.png", process_panels, columns=3)
	return result


def main() -> None:
	STACK_OUTPUT.mkdir(parents=True, exist_ok=True)
	BLEND_OUTPUT.mkdir(parents=True, exist_ok=True)

	run_figure_342()

	apple = read_rgb(IMAGE_DIR / "burt_apple.png")
	orange = read_rgb(IMAGE_DIR / "burt_orange.png")
	h, w = apple.shape[:2]
	run_blend_pair(apple, orange, "oraple_vertical", step_mask(h, w, "vertical"),
					mask_title="Vertical soft mask (white = A)")
	run_blend_pair(apple, orange, "oraple_horizontal", step_mask(h, w, "horizontal"),
					mask_title="Horizontal soft mask (white = A)")

	for label, image_a_name, image_b_name, _ in CUSTOM_PAIRS:
		image_a = read_rgb(IMAGE_DIR / image_a_name)
		image_b = read_rgb(IMAGE_DIR / image_b_name)
		height, width = image_a.shape[:2]
		mask = np.ones((height, width), dtype=np.float32)
		center = (round(0.5 * width), round(0.5 * height))
		radius = max(1, round(0.18 * min(height, width)))
		cv2.circle(mask, center, radius, 0.0, thickness=-1)
		run_blend_pair(image_a, image_b, label, mask, favorite=True,
					mask_title="Round volleyball region (black = B)")
		if label == "tennis_volleyball":
			run_blend_pair(image_a, image_b, f"{label}_vertical", step_mask(height, width),
						mask_title="Vertical soft mask (white = A)")

	galaxy = read_rgb(IMAGE_DIR / "blendPair2_galaxy.jpg")
	galaxy_mask = galaxy_silhouette_mask(galaxy)
	mask_title = "Automatic binary galaxy silhouette (white = A)"
	original_pool = read_rgb(IMAGE_DIR / "blendPair2_pool.jpg")
	original_result = run_blend_pair(galaxy, original_pool, "galaxy_pool", galaxy_mask,
								 favorite=True, mask_title=mask_title)
	gradient_pool = read_rgb(IMAGE_DIR / "swimmingpool-gradientmap.png")
	gradient_result = run_blend_pair(galaxy, gradient_pool, "galaxy_pool_gradient_map",
									 galaxy_mask, favorite=True, mask_title=mask_title)
	save_grid(BLEND_OUTPUT / "galaxy_pool_pool_comparison.png", [
		("Original pool", original_result),
		("Gradient-mapped pool", gradient_result),
	], columns=2)

	print("Part 2.3/2.4 results saved under:", OUTPUT_DIR)


if __name__ == "__main__":
	main()
