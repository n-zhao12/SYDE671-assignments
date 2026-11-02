import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from homography import build_homography_system, computeH
from mosaic import blend_mosaic
from point_picker import select_correspondences, select_points
from warp import warpImageBilinear, warpImageNearestNeighbor


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"
OUTPUT_DIR = ROOT / "output"


def _read_image(path: Path) -> np.ndarray:
	image = cv2.imread(str(path), cv2.IMREAD_COLOR)
	if image is None:
		raise FileNotFoundError(f"Could not read image: {path}")
	return image


def _save_image(path: Path, image: np.ndarray) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	output = np.clip(np.rint(image), 0, 255).astype(np.uint8)
	if not cv2.imwrite(str(path), output):
		raise OSError(f"Could not write image: {path}")


def _save_homography_system(
	path: Path,
	source_points: np.ndarray,
	destination_points: np.ndarray,
	source_key: str,
	destination_key: str,
) -> None:
	(
		system,
		values,
		normalized_source,
		normalized_destination,
		source_transform,
		destination_transform,
	) = build_homography_system(source_points, destination_points)
	parameters, _, _, _ = np.linalg.lstsq(system, values, rcond=None)
	normalized_homography = np.append(parameters, 1.0).reshape(3, 3)
	artifacts = {
		"A": system,
		"b": values,
		"normalized_source_points": normalized_source,
		"normalized_destination_points": normalized_destination,
		"source_normalization": source_transform,
		"destination_normalization": destination_transform,
		"normalized_homography": normalized_homography,
	}
	artifacts[source_key] = source_points
	artifacts[destination_key] = destination_points
	np.savez_compressed(path, **artifacts)
	np.savetxt(
		path.with_suffix(".txt"),
		np.column_stack((system, values)),
		fmt="%.10f",
		header=(
			"Each row is [A | b] for A h_normalized = b using normalized points. "
			"h_normalized = [h11, h12, h13, h21, h22, h23, h31, h32]; h33=1. "
			"Denormalize with H = inv(T_destination) @ H_normalized @ T_source."
		),
	)


def _save_source_grid(image_paths: list[Path], images: list[np.ndarray], path: Path) -> None:
	thumbnail_width = 360
	thumbnail_height = 260
	label_height = 30
	columns = min(3, len(images))
	rows = (len(images) + columns - 1) // columns
	grid = np.zeros(
		(rows * (thumbnail_height + label_height), columns * thumbnail_width, 3),
		dtype=np.uint8,
	)
	for index, (image_path, image) in enumerate(zip(image_paths, images)):
		scale = min(thumbnail_width / image.shape[1], thumbnail_height / image.shape[0])
		size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
		thumbnail = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
		row, column = divmod(index, columns)
		x = column * thumbnail_width
		y = row * (thumbnail_height + label_height)
		grid[y:y + size[1], x:x + size[0]] = thumbnail
		cv2.putText(
			grid,
			image_path.name,
			(x + 6, y + thumbnail_height + 22),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.55,
			(255, 255, 255),
			1,
		)
	_save_image(path, grid)


def _save_correspondence_view(
	moving: np.ndarray,
	reference: np.ndarray,
	moving_points: np.ndarray,
	reference_points: np.ndarray,
	path: Path,
) -> None:
	panel_width = 700
	panel_height = 700

	def fit(image):
		scale = min(panel_width / image.shape[1], panel_height / image.shape[0], 1.0)
		size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
		return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale

	moving_display, moving_scale = fit(moving)
	reference_display, reference_scale = fit(reference)
	header = 38
	gap = 12
	panel_height_actual = max(moving_display.shape[0], reference_display.shape[0])
	canvas = np.zeros((panel_height_actual + header, moving_display.shape[1] + gap + reference_display.shape[1], 3), dtype=np.uint8)
	canvas[header:header + moving_display.shape[0], :moving_display.shape[1]] = moving_display
	reference_x = moving_display.shape[1] + gap
	canvas[header:header + reference_display.shape[0], reference_x:reference_x + reference_display.shape[1]] = reference_display
	cv2.putText(canvas, "MOVING", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
	cv2.putText(canvas, "REFERENCE", (reference_x + 8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
	for index, (moving_point, reference_point) in enumerate(zip(moving_points, reference_points), start=1):
		moving_xy = (round(moving_point[0] * moving_scale), round(moving_point[1] * moving_scale) + header)
		reference_xy = (
			round(reference_point[0] * reference_scale) + reference_x,
			round(reference_point[1] * reference_scale) + header,
		)
		cv2.circle(canvas, moving_xy, 5, (0, 255, 255), -1)
		cv2.circle(canvas, reference_xy, 5, (0, 255, 255), -1)
		cv2.putText(canvas, str(index), moving_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
		cv2.putText(canvas, str(index), reference_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
		cv2.line(canvas, moving_xy, reference_xy, (0, 200, 0), 1)
	_save_image(path, canvas)


def run_rectification(image_path: Path, name: str, width: int, height: int) -> None:
	if width < 2 or height < 2:
		raise ValueError("Rectified width and height must each be at least 2 pixels")
	image = _read_image(image_path)
	source_points = select_points(
		image,
		"Click rectangle corners: top-left, top-right, bottom-right, bottom-left",
		4,
	)
	target_points = np.array([
		[0.0, 0.0],
		[width - 1.0, 0.0],
		[width - 1.0, height - 1.0],
		[0.0, height - 1.0],
	])
	homography = computeH(source_points, target_points)
	output_dir = OUTPUT_DIR / "rectification"
	output_dir.mkdir(parents=True, exist_ok=True)
	np.savetxt(output_dir / f"{name}_homography.txt", homography, fmt="%.10f")
	_save_homography_system(
		output_dir / f"{name}_linear_system.npz",
		source_points,
		target_points,
		"source_points",
		"target_points",
	)
	point_view = image.copy()
	for index, point in enumerate(source_points, start=1):
		xy = tuple(np.rint(point).astype(int))
		cv2.circle(point_view, xy, 7, (0, 255, 255), -1)
		cv2.putText(point_view, str(index), xy, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
	_save_image(output_dir / f"{name}_selected_corners.png", point_view)

	rectified_results = []
	for label, warp_function in (
		("nearest", warpImageNearestNeighbor),
		("bilinear", warpImageBilinear),
	):
		started = time.perf_counter()
		warped, valid, _ = warp_function(image, homography, (0, 0, width - 1, height - 1))
		elapsed = time.perf_counter() - started
		_save_image(output_dir / f"{name}_{label}.png", warped)
		_save_image(output_dir / f"{name}_{label}_mask.png", valid.astype(np.uint8) * 255)
		rectified_results.append(warped)
		print(f"{label} rectification: {elapsed:.3f}s; valid pixels: {valid.mean():.1%}")
	comparison = np.concatenate(rectified_results, axis=1)
	cv2.putText(comparison, "Nearest neighbor", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
	cv2.putText(
		comparison,
		"Bilinear",
		(width + 10, 28),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.8,
		(0, 0, 255),
		2,
	)
	_save_image(output_dir / f"{name}_interpolation_comparison.png", comparison)
	print(f"Rectification outputs: {output_dir}")


def run_mosaic(image_paths: list[Path], name: str, reselect: bool = False) -> None:
	if len(image_paths) < 2:
		raise ValueError("A mosaic requires at least two images")
	images = [_read_image(path) for path in image_paths]
	output_dir = OUTPUT_DIR / "mosaics"
	correspondence_dir = OUTPUT_DIR / "correspondences"
	correspondence_dir.mkdir(parents=True, exist_ok=True)

	homographies = [np.eye(3, dtype=np.float64)]
	for index in range(1, len(images)):
		cache_path = correspondence_dir / f"{name}_pair_{index:02}.npz"
		if cache_path.exists() and not reselect:
			with np.load(cache_path) as saved:
				moving_points = saved["moving_points"]
				reference_points = saved["reference_points"]
		else:
			moving_points, reference_points = select_correspondences(
				images[index],
				images[index - 1],
				f"{name}: image {index + 1} to image {index}",
			)
			np.savez_compressed(
				cache_path,
				moving_points=moving_points,
				reference_points=reference_points,
			)
		pair_homography = computeH(moving_points, reference_points)
		homographies.append(homographies[-1] @ pair_homography)
		_save_correspondence_view(
			images[index],
			images[index - 1],
			moving_points,
			reference_points,
			correspondence_dir / f"{name}_pair_{index:02}_matches.png",
		)
		np.savetxt(
			correspondence_dir / f"{name}_pair_{index:02}_homography.txt",
			pair_homography,
			fmt="%.10f",
		)
		_save_homography_system(
			correspondence_dir / f"{name}_pair_{index:02}_linear_system.npz",
			moving_points,
			reference_points,
			"moving_points",
			"reference_points",
		)
		print(f"Registered image {index + 1}/{len(images)} using {len(moving_points)} point pairs")

	mosaic, valid, bounds = blend_mosaic(images, homographies)
	output_dir.mkdir(parents=True, exist_ok=True)
	_save_source_grid(image_paths, images, output_dir / f"{name}_sources.png")
	_save_image(output_dir / f"{name}_mosaic.png", mosaic)
	_save_image(output_dir / f"{name}_mask.png", valid.astype(np.uint8) * 255)
	np.savez_compressed(output_dir / f"{name}_transforms.npz", homographies=np.asarray(homographies), bounds=bounds)
	print(f"Mosaic saved: {output_dir / f'{name}_mosaic.png'}")
	print(f"Canvas bounds in reference coordinates: {bounds}")


def _images_for_set(set_number: str) -> list[Path]:
	pattern = f"imageSet{set_number}_*"
	images = [
		path for path in DATA_DIR.glob(pattern)
		if path.suffix.lower() in (".jpg", ".jpeg", ".png")
		and path.stem.rsplit("_", 1)[-1].isdigit()
	]
	images.sort(key=lambda path: int(path.stem.rsplit("_", 1)[-1]))
	if len(images) < 2:
		raise FileNotFoundError(f"Expected at least two images matching {DATA_DIR / pattern}")
	return images


def main() -> None:
	parser = argparse.ArgumentParser(description="Assignment 3 Part 1: manual homographies, warping, and mosaics")
	subparsers = parser.add_subparsers(dest="command", required=True)

	rectify_parser = subparsers.add_parser("rectify", help="Rectify a known rectangular object")
	rectify_parser.add_argument("image", type=Path)
	rectify_parser.add_argument("--name", default=None)
	rectify_parser.add_argument("--width", type=int, default=800)
	rectify_parser.add_argument("--height", type=int, default=600)

	mosaic_parser = subparsers.add_parser("mosaic", help="Manually match and blend an ordered image sequence")
	mosaic_parser.add_argument("--name", default=None)
	image_source = mosaic_parser.add_mutually_exclusive_group(required=True)
	image_source.add_argument("--set", choices=("1", "2", "3"), help="Load imageSetN_* from Assignment3/data")
	image_source.add_argument("--images", type=Path, nargs="+", help="Explicit ordered image paths")
	mosaic_parser.add_argument("--reselect", action="store_true", help="Ignore saved correspondences and click them again")

	args = parser.parse_args()
	if args.command == "rectify":
		run_rectification(args.image, args.name or args.image.stem, args.width, args.height)
	elif args.command == "mosaic":
		if args.set is not None:
			image_paths = _images_for_set(args.set)
			name = args.name or f"set{args.set}"
		else:
			image_paths = args.images
			name = args.name or "mosaic"
		run_mosaic(image_paths, name, args.reselect)


if __name__ == "__main__":
	main()
