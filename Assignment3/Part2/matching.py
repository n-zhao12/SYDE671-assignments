from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PART1_DIR = Path(__file__).resolve().parents[1] / "Part1"
if str(PART1_DIR) not in sys.path:
	sys.path.insert(0, str(PART1_DIR))

from homography import computeH, project_points


def match_descriptors(
	descriptors_a: np.ndarray,
	descriptors_b: np.ndarray,
	ratio_threshold: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
	"""Match descriptors using Lowe's nearest/second-nearest distance ratio.

	Returns (index_pairs, ratios), with index_pairs columns indexing A and B.
	Matches are made one-to-one by retaining the lowest-ratio match per B feature.
	"""
	a = np.asarray(descriptors_a, dtype=np.float64)
	b = np.asarray(descriptors_b, dtype=np.float64)
	if a.ndim != 2 or b.ndim != 2 or a.shape[1:] != b.shape[1:]:
		raise ValueError("descriptor arrays must be 2D with matching descriptor sizes")
	if not 0 < ratio_threshold < 1:
		raise ValueError("ratio_threshold must be between 0 and 1")
	if len(a) == 0 or len(b) < 2:
		return np.empty((0, 2), dtype=np.intp), np.empty(0, dtype=np.float64)
	if not np.isfinite(a).all() or not np.isfinite(b).all():
		raise ValueError("descriptors must be finite")

	squared_distances = (
		np.sum(a * a, axis=1)[:, None]
		+ np.sum(b * b, axis=1)[None, :]
		- 2.0 * (a @ b.T)
	)
	np.maximum(squared_distances, 0.0, out=squared_distances)
	nearest_two = np.argpartition(squared_distances, kth=1, axis=1)[:, :2]
	nearest_distances = np.take_along_axis(squared_distances, nearest_two, axis=1)
	order = np.argsort(nearest_distances, axis=1)
	nearest_two = np.take_along_axis(nearest_two, order, axis=1)
	nearest_distances = np.take_along_axis(nearest_distances, order, axis=1)
	first_distance = np.sqrt(nearest_distances[:, 0])
	second_distance = np.sqrt(nearest_distances[:, 1])
	ratios = np.divide(
		first_distance,
		second_distance,
		out=np.full_like(first_distance, np.inf),
		where=second_distance > np.finfo(np.float64).eps,
	)
	accepted = np.flatnonzero(ratios < ratio_threshold)
	if len(accepted) == 0:
		return np.empty((0, 2), dtype=np.intp), np.empty(0, dtype=np.float64)
	accepted = accepted[np.argsort(ratios[accepted])]
	used_b: set[int] = set()
	unique = []
	for index_a in accepted:
		index_b = int(nearest_two[index_a, 0])
		if index_b not in used_b:
			used_b.add(index_b)
			unique.append(index_a)
	unique = np.asarray(unique, dtype=np.intp)
	return np.column_stack((unique, nearest_two[unique, 0])), ratios[unique]


def ransac_homography(
	source_points: np.ndarray,
	destination_points: np.ndarray,
	threshold: float = 4.0,
	max_iterations: int = 2000,
	confidence: float = 0.995,
	min_iterations: int = 100,
	seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Estimate source-to-destination H with four-point RANSAC from scratch."""
	source = np.asarray(source_points, dtype=np.float64)
	destination = np.asarray(destination_points, dtype=np.float64)
	if source.ndim != 2 or source.shape[1] != 2 or destination.shape != source.shape:
		raise ValueError("source_points and destination_points must both have shape (n, 2)")
	if len(source) < 4:
		raise ValueError("RANSAC requires at least four point correspondences")
	if not np.isfinite(source).all() or not np.isfinite(destination).all():
		raise ValueError("point correspondences must be finite")
	if threshold <= 0 or max_iterations < 1 or min_iterations < 1:
		raise ValueError("threshold and iteration counts must be positive")
	if not 0 < confidence < 1:
		raise ValueError("confidence must be between 0 and 1")

	rng = np.random.default_rng(seed)
	best_inliers = np.zeros(len(source), dtype=bool)
	best_median_error = np.inf
	iteration_limit = max_iterations
	iteration = 0
	while iteration < iteration_limit:
		iteration += 1
		sample = rng.choice(len(source), size=4, replace=False)
		if np.linalg.matrix_rank(source[sample] - source[sample].mean(axis=0)) < 2:
			continue
		if np.linalg.matrix_rank(destination[sample] - destination[sample].mean(axis=0)) < 2:
			continue
		try:
			homography = computeH(source[sample], destination[sample])
			projected = project_points(source, homography)
		except (ValueError, np.linalg.LinAlgError):
			continue
		errors = np.linalg.norm(projected - destination, axis=1)
		inliers = errors <= threshold
		inlier_count = int(np.count_nonzero(inliers))
		if inlier_count < 4:
			continue
		median_error = float(np.median(errors[inliers]))
		if inlier_count > np.count_nonzero(best_inliers) or (
			inlier_count == np.count_nonzero(best_inliers)
			and median_error < best_median_error
		):
			best_inliers = inliers
			best_median_error = median_error
			inlier_ratio = inlier_count / len(source)
			all_inlier_probability = inlier_ratio ** 4
			if all_inlier_probability >= 1.0:
				required = min_iterations
			elif all_inlier_probability > 0:
				required = int(np.ceil(np.log(1.0 - confidence) / np.log(1.0 - all_inlier_probability)))
			else:
				required = max_iterations
			iteration_limit = min(max_iterations, max(min_iterations, required))

	if np.count_nonzero(best_inliers) < 4:
		raise ValueError("RANSAC found fewer than four inliers; check feature matches")

	for _ in range(2):
		homography = computeH(source[best_inliers], destination[best_inliers])
		projected = project_points(source, homography)
		errors = np.linalg.norm(projected - destination, axis=1)
		refined_inliers = errors <= threshold
		if np.count_nonzero(refined_inliers) < 4 or np.array_equal(refined_inliers, best_inliers):
			break
		best_inliers = refined_inliers
	homography = computeH(source[best_inliers], destination[best_inliers])
	projected = project_points(source, homography)
	errors = np.linalg.norm(projected - destination, axis=1)
	best_inliers = errors <= threshold
	return homography, best_inliers, errors