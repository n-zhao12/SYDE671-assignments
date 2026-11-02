from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PART1_DIR = Path(__file__).resolve().parents[1] / "Part1"
if str(PART1_DIR) not in sys.path:
	sys.path.insert(0, str(PART1_DIR))

from mosaic import blend_mosaic

from features import detect_and_describe
from matching import match_descriptors, ransac_homography


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"
OUTPUT_DIR = ROOT / "output"
MANUAL_MOSAIC_DIR = PART1_DIR / "output" / "mosaics"


def _read_resized(path: Path, max_side: int) -> np.ndarray:
	image = cv2.imread(str(path), cv2.IMREAD_COLOR)
	if image is None:
		raise FileNotFoundError(f"Could not read image: {path}")
	height, width = image.shape[:2]
	scale = min(1.0, max_side / max(height, width))
	if scale < 1.0:
		image = cv2.resize(
			image,
			(max(1, round(width * scale)), max(1, round(height * scale))),
			interpolation=cv2.INTER_AREA,
		)
	return image


def _save_image(path: Path, image: np.ndarray) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	output = np.clip(np.rint(image), 0, 255).astype(np.uint8)
	if not cv2.imwrite(str(path), output):
		raise OSError(f"Could not write image: {path}")


def _save_keypoint_view(
	image: np.ndarray, points_xy: np.ndarray, path: Path, color: tuple[int, int, int]
) -> None:
	view = image.copy()
	for x, y in points_xy:
		cv2.circle(view, (round(x), round(y)), 2, color, -1, lineType=cv2.LINE_AA)
	_save_image(path, view)


def _save_descriptor_grid(patches: np.ndarray, path: Path, limit: int = 25) -> None:
	if len(patches) == 0:
		return
	selected = patches[:limit]
	tile_size = 96
	label_height = 22
	columns = min(5, len(selected))
	rows = (len(selected) + columns - 1) // columns
	canvas = np.zeros((rows * (tile_size + label_height), columns * tile_size, 3), dtype=np.uint8)
	for index, patch in enumerate(selected):
		row, column = divmod(index, columns)
		x, y = column * tile_size, row * (tile_size + label_height)
		patch_u8 = np.clip(patch * 255.0, 0, 255).astype(np.uint8)
		tile = cv2.resize(patch_u8, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
		canvas[y:y + tile_size, x:x + tile_size] = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
		cv2.putText(canvas, f"descriptor {index + 1}", (x + 3, y + tile_size + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
	_save_image(path, canvas)


def _fit_panel(image: np.ndarray, points: np.ndarray, max_width: int, max_height: int):
	height, width = image.shape[:2]
	scale = min(max_width / width, max_height / height, 1.0)
	resized = cv2.resize(
		image,
		(max(1, round(width * scale)), max(1, round(height * scale))),
		interpolation=cv2.INTER_AREA,
	)
	return resized, points * scale, scale


def _save_match_view(
	image_a: np.ndarray,
	points_a: np.ndarray,
	image_b: np.ndarray,
	points_b: np.ndarray,
	matches: np.ndarray,
	path: Path,
	max_lines: int = 250,
) -> None:
	panel_width, panel_height, header = 850, 700, 36
	view_a, scaled_a, _ = _fit_panel(image_a, points_a, panel_width, panel_height)
	view_b, scaled_b, _ = _fit_panel(image_b, points_b, panel_width, panel_height)
	gap = 12
	panel_h = max(view_a.shape[0], view_b.shape[0])
	panel_w = view_a.shape[1] + gap + view_b.shape[1]
	canvas = np.zeros((panel_h + header, panel_w, 3), dtype=np.uint8)
	canvas[header:header + view_a.shape[0], :view_a.shape[1]] = view_a
	x_b = view_a.shape[1] + gap
	canvas[header:header + view_b.shape[0], x_b:x_b + view_b.shape[1]] = view_b
	cv2.putText(canvas, "IMAGE A", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
	cv2.putText(canvas, "IMAGE B", (x_b + 8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
	if len(matches) > max_lines:
		indices = np.linspace(0, len(matches) - 1, max_lines, dtype=np.intp)
		matches = matches[indices]
	for index_a, index_b in matches:
		point_a = tuple(np.rint(scaled_a[index_a]).astype(int) + np.array([0, header]))
		point_b = tuple(np.rint(scaled_b[index_b]).astype(int) + np.array([x_b, header]))
		color = tuple(int(value) for value in np.random.default_rng(int(index_a + index_b)).integers(70, 256, size=3))
		cv2.circle(canvas, point_a, 4, color, -1, lineType=cv2.LINE_AA)
		cv2.circle(canvas, point_b, 4, color, -1, lineType=cv2.LINE_AA)
		cv2.line(canvas, point_a, point_b, color, 1, lineType=cv2.LINE_AA)
	_save_image(path, canvas)


def _save_source_grid(images: list[np.ndarray], paths: list[Path], destination: Path) -> None:
	thumb_width, thumb_height, label_height = 360, 260, 28
	columns = min(3, len(images))
	rows = (len(images) + columns - 1) // columns
	canvas = np.zeros((rows * (thumb_height + label_height), columns * thumb_width, 3), dtype=np.uint8)
	for index, (image, path) in enumerate(zip(images, paths)):
		height, width = image.shape[:2]
		scale = min(thumb_width / width, thumb_height / height)
		size = (max(1, round(width * scale)), max(1, round(height * scale)))
		thumbnail = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
		row, column = divmod(index, columns)
		x, y = column * thumb_width, row * (thumb_height + label_height)
		canvas[y:y + size[1], x:x + size[0]] = thumbnail
		cv2.putText(canvas, path.name, (x + 5, y + thumb_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
	_save_image(destination, canvas)


def _save_manual_comparison(automatic_path: Path, manual_path: Path, destination: Path) -> None:
	if not manual_path.exists():
		print(f"Manual mosaic not found; skipping comparison: {manual_path}")
		return
	automatic = cv2.imread(str(automatic_path), cv2.IMREAD_COLOR)
	manual = cv2.imread(str(manual_path), cv2.IMREAD_COLOR)
	if automatic is None or manual is None:
		print(f"Could not read a mosaic for side-by-side comparison: {automatic_path}, {manual_path}")
		return
	target_height = 850
	panels = []
	for image, label in ((manual, "MANUAL"), (automatic, "AUTOMATIC")):
		scale = target_height / image.shape[0]
		width = max(1, round(image.shape[1] * scale))
		panel = cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
		cv2.putText(panel, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
		panels.append(panel)
	_save_image(destination, np.concatenate(panels, axis=1))


def _images_for_set(set_number: str) -> list[Path]:
	paths = [
		path for path in DATA_DIR.glob(f"imageSet{set_number}_*")
		if path.suffix.lower() in (".jpg", ".jpeg", ".png")
		and path.stem.rsplit("_", 1)[-1].isdigit()
	]
	paths.sort(key=lambda path: int(path.stem.rsplit("_", 1)[-1]))
	if len(paths) < 2:
		raise FileNotFoundError(f"Expected at least two images for set {set_number} in {DATA_DIR}")
	return paths


def run_set(
	set_number: str,
	max_side: int = 1200,
	max_features: int = 500,
	candidate_limit: int = 5000,
	ratio_threshold: float = 0.8,
	transform_threshold: float = 4.0,
	seed: int = 0,
) -> None:
	paths = _images_for_set(set_number)
	images = [_read_resized(path, max_side) for path in paths]
	grayscale = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 for image in images]
	set_dir = OUTPUT_DIR / f"set{set_number}"
	feature_dir = set_dir / "features"
	match_dir = set_dir / "matches"
	mosaic_dir = set_dir / "mosaics"
	for directory in (feature_dir, match_dir, mosaic_dir):
		directory.mkdir(parents=True, exist_ok=True)

	features = []
	for index, (path, image, gray) in enumerate(zip(paths, images, grayscale), start=1):
		result = detect_and_describe(gray, max_features=max_features, candidate_limit=candidate_limit)
		features.append(result)
		stem = f"image{index:02}"
		_save_keypoint_view(image, result["corners_xy"], feature_dir / f"{stem}_harris.png", (0, 255, 255))
		_save_keypoint_view(image, result["anms_xy"], feature_dir / f"{stem}_anms.png", (0, 255, 0))
		_save_descriptor_grid(result["patches"], feature_dir / f"{stem}_descriptors.png")
		np.savez_compressed(
			feature_dir / f"{stem}_features.npz",
			corners_xy=result["corners_xy"],
			corner_strengths=result["corner_strengths"],
			anms_xy=result["anms_xy"],
			anms_strengths=result["anms_strengths"],
			points_xy=result["points_xy"],
			descriptors=result["descriptors"],
			patches=result["patches"],
		)
		print(
			f"Set {set_number}, image {index}/{len(images)}: "
			f"{len(result['corners_xy'])} Harris candidates, "
			f"{len(result['points_xy'])} descriptors"
		)

	homographies = [np.eye(3, dtype=np.float64)]
	for index in range(1, len(images)):
		previous = features[index - 1]
		current = features[index]
		matches, ratios = match_descriptors(
			current["descriptors"], previous["descriptors"], ratio_threshold
		)
		if len(matches) < 4:
			raise ValueError(f"Set {set_number} pair {index}: only {len(matches)} ratio-test matches")
		source_points = current["points_xy"][matches[:, 0]]
		destination_points = previous["points_xy"][matches[:, 1]]
		pair_seed = seed + int(set_number) * 100 + index
		homography, inliers, errors = ransac_homography(
			source_points,
			destination_points,
			threshold=transform_threshold,
			seed=pair_seed,
		)
		homographies.append(homographies[-1] @ homography)
		stem = f"pair{index:02}"
		_save_match_view(
			images[index],
			current["points_xy"],
			images[index - 1],
			previous["points_xy"],
			matches,
			match_dir / f"{stem}_matches.png",
		)
		_save_match_view(
			images[index],
			current["points_xy"],
			images[index - 1],
			previous["points_xy"],
			matches[inliers],
			match_dir / f"{stem}_inliers.png",
		)
		match_data = {
			"matches": matches,
			"ratios": ratios,
			"inliers": inliers,
			"errors": errors,
			"source_points": source_points,
			"destination_points": destination_points,
			"homography": homography,
		}
		np.savez_compressed(match_dir / f"{stem}_data.npz", **match_data)
		np.savetxt(match_dir / f"{stem}_homography.txt", homography, fmt="%.10f")
		print(
			f"Set {set_number}, pair {index}: {len(matches)} ratio matches, "
			f"{int(np.count_nonzero(inliers))} RANSAC inliers, "
			f"median inlier error {np.median(errors[inliers]):.2f}px"
		)

	automatic, valid, bounds = blend_mosaic(images, homographies)
	automatic_path = mosaic_dir / f"set{set_number}_automatic.png"
	_save_image(automatic_path, automatic)
	_save_image(mosaic_dir / f"set{set_number}_mask.png", valid.astype(np.uint8) * 255)
	_save_source_grid(images, paths, mosaic_dir / f"set{set_number}_sources.png")
	np.savez_compressed(mosaic_dir / f"set{set_number}_transforms.npz", homographies=np.asarray(homographies), bounds=bounds)
	manual_path = MANUAL_MOSAIC_DIR / f"set{set_number}_mosaic.png"
	_save_manual_comparison(
		automatic_path,
		manual_path,
		mosaic_dir / f"set{set_number}_manual_vs_automatic.png",
	)
	print(f"Set {set_number} automatic mosaic saved: {automatic_path}; bounds={bounds}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Assignment 3 Part 2: automatic feature matching and stitching")
	parser.add_argument("--set", choices=("all", "1", "2", "3"), default="all")
	parser.add_argument("--max-side", type=int, default=1200, help="Resize images to this longest side for feature extraction")
	parser.add_argument("--features", type=int, default=500)
	parser.add_argument("--candidates", type=int, default=5000)
	parser.add_argument("--ratio", type=float, default=0.8)
	parser.add_argument("--ransac-threshold", type=float, default=4.0)
	parser.add_argument("--seed", type=int, default=0)
	args = parser.parse_args()
	if args.max_side < 100 or args.features < 4 or args.candidates < args.features:
		parser.error("require --max-side >= 100, --features >= 4, and --candidates >= --features")
	sets = ("1", "2", "3") if args.set == "all" else (args.set,)
	for set_number in sets:
		run_set(
			set_number,
			max_side=args.max_side,
			max_features=args.features,
			candidate_limit=args.candidates,
			ratio_threshold=args.ratio,
			transform_threshold=args.ransac_threshold,
			seed=args.seed,
		)


if __name__ == "__main__":
	main()