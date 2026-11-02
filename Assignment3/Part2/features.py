from __future__ import annotations

import cv2
import numpy as np

from harris import get_harris_corners


def adaptive_non_maximal_suppression(
	points_xy: np.ndarray,
	strengths: np.ndarray,
	max_points: int = 500,
	robustness: float = 0.9,
	chunk_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
	"""Choose spatially distributed corners using the MOPS suppression radius."""
	points = np.asarray(points_xy, dtype=np.float64)
	strengths = np.asarray(strengths, dtype=np.float64)
	if points.ndim != 2 or points.shape[1] != 2 or strengths.shape != (len(points),):
		raise ValueError("points_xy must be (n, 2) and strengths must be (n,)")
	if not 0 < robustness < 1:
		raise ValueError("robustness must be between 0 and 1")
	if max_points < 0 or chunk_size < 1:
		raise ValueError("max_points must be non-negative and chunk_size positive")
	if len(points) == 0 or max_points == 0:
		return points[:0].copy(), np.empty(0, dtype=np.float64)
	if not np.isfinite(points).all() or not np.isfinite(strengths).all():
		raise ValueError("points and strengths must be finite")
	if np.any(strengths <= 0):
		raise ValueError("Harris strengths must be positive for ANMS")

	count = len(points)
	radii_squared = np.full(count, np.inf, dtype=np.float64)
	for start in range(0, count, chunk_size):
		stop = min(start + chunk_size, count)
		delta = points[start:stop, None, :] - points[None, :, :]
		distances_squared = np.sum(delta * delta, axis=2)
		stronger = strengths[None, :] > strengths[start:stop, None] / robustness
		distances_squared[~stronger] = np.inf
		radii_squared[start:stop] = np.min(distances_squared, axis=1)

	order = np.argsort(radii_squared)[::-1][:min(max_points, count)]
	return points[order].copy(), np.sqrt(radii_squared[order])


def _bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
	height, width = image.shape
	x0 = np.floor(x).astype(np.intp)
	y0 = np.floor(y).astype(np.intp)
	x1 = np.minimum(x0 + 1, width - 1)
	y1 = np.minimum(y0 + 1, height - 1)
	dx = x - x0
	dy = y - y0
	top = (1.0 - dx) * image[y0, x0] + dx * image[y0, x1]
	bottom = (1.0 - dx) * image[y1, x0] + dx * image[y1, x1]
	return (1.0 - dy) * top + dy * bottom


def extract_descriptors(
	image_gray: np.ndarray,
	points_xy: np.ndarray,
	window_size: int = 40,
	descriptor_size: int = 8,
	gaussian_sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Extract blurred, axis-aligned, bias/gain-normalized 8x8 descriptors.

	The 8x8 samples span a 40x40 neighborhood. Returns only points with complete
	neighborhoods and non-constant descriptors, their normalized descriptors,
	and the corresponding unnormalized 8x8 samples for visualization.
	"""
	image = np.asarray(image_gray, dtype=np.float32)
	points = np.asarray(points_xy, dtype=np.float64)
	if image.ndim != 2 or image.size == 0:
		raise ValueError("image_gray must be a non-empty grayscale image")
	if points.ndim != 2 or points.shape[1] != 2:
		raise ValueError("points_xy must have shape (n, 2)")
	if window_size < descriptor_size or descriptor_size < 2 or gaussian_sigma < 0:
		raise ValueError("invalid descriptor window, size, or blur sigma")
	if window_size % descriptor_size != 0:
		raise ValueError("window_size must be divisible by descriptor_size")

	blurred = cv2.GaussianBlur(image, (0, 0), gaussian_sigma)
	sample_spacing = window_size / descriptor_size
	offset = (np.arange(descriptor_size, dtype=np.float64) - (descriptor_size - 1) / 2.0)
	offset *= sample_spacing
	grid_x, grid_y = np.meshgrid(offset, offset)
	margin = window_size / 2.0
	valid_points = []
	descriptors = []
	patches = []
	for x, y in points:
		if x < margin or x > image.shape[1] - 1 - margin:
			continue
		if y < margin or y > image.shape[0] - 1 - margin:
			continue
		samples = _bilinear_sample(blurred, x + grid_x, y + grid_y)
		mean = float(np.mean(samples))
		centered = samples - mean
		standard_deviation = float(np.std(centered))
		if standard_deviation < 1e-6:
			continue
		valid_points.append((x, y))
		patches.append(samples)
		descriptors.append(centered / standard_deviation)

	if not descriptors:
		return (
			np.empty((0, 2), dtype=np.float64),
			np.empty((0, descriptor_size * descriptor_size), dtype=np.float32),
			np.empty((0, descriptor_size, descriptor_size), dtype=np.float32),
		)
	return (
		np.asarray(valid_points, dtype=np.float64),
		np.asarray(descriptors, dtype=np.float32).reshape(-1, descriptor_size * descriptor_size),
		 np.asarray(patches, dtype=np.float32),
	)


def detect_and_describe(
	image_gray: np.ndarray,
	max_features: int = 500,
	candidate_limit: int = 5000,
	edge_discard: int = 20,
) -> dict[str, np.ndarray]:
	"""Run single-scale Harris, ANMS, and MOPS-style descriptor extraction."""
	image = np.asarray(image_gray, dtype=np.float32)
	if image.ndim != 2:
		raise ValueError("image_gray must be grayscale")
	harris_response, coordinates_yx = get_harris_corners(image, edge_discard=edge_discard)
	if coordinates_yx.size == 0:
		return {
			"harris": harris_response,
			"corners_xy": np.empty((0, 2), dtype=np.float64),
			"corner_strengths": np.empty(0, dtype=np.float64),
			"anms_xy": np.empty((0, 2), dtype=np.float64),
			"anms_strengths": np.empty(0, dtype=np.float64),
			"points_xy": np.empty((0, 2), dtype=np.float64),
			"descriptors": np.empty((0, 64), dtype=np.float32),
			"patches": np.empty((0, 8, 8), dtype=np.float32),
		}

	strengths = harris_response[coordinates_yx[0], coordinates_yx[1]].astype(np.float64)
	positive = strengths > 0
	coordinates_yx = coordinates_yx[:, positive]
	strengths = strengths[positive]
	if len(strengths) > candidate_limit:
		keep = np.argpartition(strengths, -candidate_limit)[-candidate_limit:]
		coordinates_yx = coordinates_yx[:, keep]
		strengths = strengths[keep]
	corners_xy = coordinates_yx[[1, 0]].T.astype(np.float64)
	anms_xy, anms_radii = adaptive_non_maximal_suppression(
		corners_xy, strengths, max_points=max_features
	)
	strength_by_point = {tuple(point): strength for point, strength in zip(corners_xy, strengths)}
	anms_strengths = np.asarray([strength_by_point[tuple(point)] for point in anms_xy])
	points_xy, descriptors, patches = extract_descriptors(image, anms_xy)
	return {
		"harris": harris_response,
		"corners_xy": corners_xy,
		"corner_strengths": strengths,
		"anms_xy": anms_xy,
		"anms_strengths": anms_strengths,
		"anms_radii": anms_radii,
		"points_xy": points_xy,
		"descriptors": descriptors,
		"patches": patches,
	}