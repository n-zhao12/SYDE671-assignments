import numpy as np


def _project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
	homogeneous = np.column_stack((points, np.ones(points.shape[0])))
	projected = (homography @ homogeneous.T).T
	denominator = projected[:, 2]
	if np.any(np.isclose(denominator, 0.0)) or not (
		 np.all(denominator > 0) or np.all(denominator < 0)
	):
		raise ValueError("The transform sends image corners to or across infinity")
	return projected[:, :2] / denominator[:, None]


def _bounds_for_image(image_shape: tuple[int, ...], homography: np.ndarray):
	height, width = image_shape[:2]
	corners = np.array([
		[0.0, 0.0],
		[width - 1.0, 0.0],
		[width - 1.0, height - 1.0],
		[0.0, height - 1.0],
	])
	projected = _project_points(corners, homography)
	nearest_integer = np.rint(projected)
	projected = np.where(np.isclose(projected, nearest_integer, atol=1e-8, rtol=0), nearest_integer, projected)
	minimum = np.floor(np.min(projected, axis=0)).astype(int)
	maximum = np.ceil(np.max(projected, axis=0)).astype(int)
	return int(minimum[0]), int(minimum[1]), int(maximum[0]), int(maximum[1])


def warp_image(
	image: np.ndarray,
	homography: np.ndarray,
	interpolation: str = "bilinear",
	output_bounds: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
	"""Inverse-warp an image; H maps source (x, y) into output/world coordinates.

	Returns the warped float image, a valid-pixel mask, and the world coordinate
	represented by output pixel (0, 0). Bounds are (min_x, min_y, max_x, max_y).
	"""
	image = np.asarray(image)
	homography = np.asarray(homography, dtype=np.float64)
	if image.ndim not in (2, 3) or image.shape[0] == 0 or image.shape[1] == 0:
		raise ValueError("image must be a non-empty 2D or 3D array")
	if homography.shape != (3, 3) or not np.isfinite(homography).all():
		raise ValueError("homography must be a finite 3x3 matrix")
	if interpolation not in ("nearest", "bilinear"):
		raise ValueError("interpolation must be 'nearest' or 'bilinear'")

	if output_bounds is None:
		output_bounds = _bounds_for_image(image.shape, homography)
	min_x, min_y, max_x, max_y = output_bounds
	if max_x < min_x or max_y < min_y:
		raise ValueError("output_bounds must have increasing coordinates")

	output_height = max_y - min_y + 1
	output_width = max_x - min_x + 1
	x_world, y_world = np.meshgrid(
		np.arange(min_x, max_x + 1, dtype=np.float64),
		np.arange(min_y, max_y + 1, dtype=np.float64),
	)
	destination = np.stack((x_world.ravel(), y_world.ravel(), np.ones(x_world.size)))
	source = np.linalg.inv(homography) @ destination
	denominator = source[2]
	finite = np.isfinite(source).all(axis=0) & ~np.isclose(denominator, 0.0)
	source_x = np.zeros_like(denominator)
	source_y = np.zeros_like(denominator)
	source_x[finite] = source[0, finite] / denominator[finite]
	source_y[finite] = source[1, finite] / denominator[finite]

	height, width = image.shape[:2]
	channels = () if image.ndim == 2 else (image.shape[2],)
	result = np.zeros((output_height * output_width,) + channels, dtype=np.float64)
	if interpolation == "nearest":
		valid = finite & (source_x >= 0) & (source_x <= width - 1) & \
			(source_y >= 0) & (source_y <= height - 1)
		x_index = np.rint(source_x[valid]).astype(int)
		y_index = np.rint(source_y[valid]).astype(int)
		result[valid] = image[y_index, x_index]
	else:
		valid = finite & (source_x >= 0) & (source_x <= width - 1) & \
			(source_y >= 0) & (source_y <= height - 1)
		x0 = np.floor(source_x[valid]).astype(int)
		y0 = np.floor(source_y[valid]).astype(int)
		x1 = np.minimum(x0 + 1, width - 1)
		y1 = np.minimum(y0 + 1, height - 1)
		dx = (source_x[valid] - x0).reshape((-1,) + (1,) * len(channels))
		dy = (source_y[valid] - y0).reshape((-1,) + (1,) * len(channels))
		top = (1.0 - dx) * image[y0, x0] + dx * image[y0, x1]
		bottom = (1.0 - dx) * image[y1, x0] + dx * image[y1, x1]
		result[valid] = (1.0 - dy) * top + dy * bottom

	return (
		result.reshape((output_height, output_width) + channels),
		valid.reshape(output_height, output_width),
		(min_x, min_y),
	)


def warpImageNearestNeighbor(
	image: np.ndarray,
	homography: np.ndarray,
	output_bounds: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
	return warp_image(image, homography, "nearest", output_bounds)


def warpImageBilinear(
	image: np.ndarray,
	homography: np.ndarray,
	output_bounds: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
	return warp_image(image, homography, "bilinear", output_bounds)