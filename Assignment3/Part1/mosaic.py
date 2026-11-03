import numpy as np

from warp import warp_image


def transformed_bounds(
    image_shape: tuple[int, ...], homography: np.ndarray
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    corners = np.array([
        [0.0, 0.0],
        [width - 1.0, 0.0],
        [width - 1.0, height - 1.0],
        [0.0, height - 1.0],
    ])
    projected = (homography @ np.column_stack((corners, np.ones(4))).T).T
    denominator = projected[:, 2]
    if np.any(np.isclose(denominator, 0.0)) or not (
        np.all(denominator > 0) or np.all(denominator < 0)
    ):
        raise ValueError("The transform sends image corners to or across infinity")
    projected = projected[:, :2] / denominator[:, None]
    nearest_integer = np.rint(projected)
    projected = np.where(np.isclose(projected, nearest_integer, atol=1e-8, rtol=0), nearest_integer, projected)
    minimum = np.floor(np.min(projected, axis=0)).astype(int)
    maximum = np.ceil(np.max(projected, axis=0)).astype(int)
    return int(minimum[0]), int(minimum[1]), int(maximum[0]), int(maximum[1])


def feather_weight_map(height: int, width: int) -> np.ndarray:
    """Return normalized edge-distance weights for feather blending."""
    if height < 1 or width < 1:
        raise ValueError("feather weight dimensions must be positive")
    y, x = np.indices((height, width), dtype=np.float64)
    distance_to_edge = np.minimum.reduce((x + 1, y + 1, width - x, height - y))
    return distance_to_edge / max(1.0, float(distance_to_edge.max()))


def blend_mosaic(
    images: list[np.ndarray], homographies: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Warp images into a shared world canvas and feather-average overlaps.

    Each homography maps its image's pixel coordinates into the same world frame.
    """
    if not images or len(images) != len(homographies):
        raise ValueError("Provide the same non-zero number of images and homographies")
    channels = images[0].shape[2:] if images[0].ndim == 3 else ()
    if any((image.shape[2:] if image.ndim == 3 else ()) != channels for image in images):
        raise ValueError("All images must have the same number of channels")

    bounds = [transformed_bounds(image.shape, matrix) for image, matrix in zip(images, homographies)]
    canvas_bounds = (
        min(bound[0] for bound in bounds),
        min(bound[1] for bound in bounds),
        max(bound[2] for bound in bounds),
        max(bound[3] for bound in bounds),
    )
    min_x, min_y, max_x, max_y = canvas_bounds
    canvas_shape = (max_y - min_y + 1, max_x - min_x + 1) + channels
    if canvas_shape[0] * canvas_shape[1] > 100_000_000:
        raise ValueError("Projected mosaic canvas is too large to allocate")

    total = np.zeros(canvas_shape, dtype=np.float64)
    weight_sum = np.zeros(canvas_shape[:2], dtype=np.float64)
    for image, homography in zip(images, homographies):
        warped, valid, _ = warp_image(image, homography, "bilinear", canvas_bounds)
        source_weights = feather_weight_map(image.shape[0], image.shape[1])
        warped_weights, weight_valid, _ = warp_image(
            source_weights, homography, "bilinear", canvas_bounds
        )
        weights = np.where(valid & weight_valid, warped_weights, 0.0)
        if channels:
            total += warped * weights[..., None]
        else:
            total += warped * weights
        weight_sum += weights

    valid_mosaic = weight_sum > 0
    if channels:
        mosaic = np.divide(
            total,
            weight_sum[..., None],
            out=np.zeros_like(total),
            where=weight_sum[..., None] > 0,
        )
    else:
        mosaic = np.divide(total, weight_sum, out=np.zeros_like(total), where=weight_sum > 0)
    return mosaic, valid_mosaic, canvas_bounds