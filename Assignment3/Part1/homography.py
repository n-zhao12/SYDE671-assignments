import numpy as np


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    mean_distance = np.mean(np.linalg.norm(centered, axis=1))
    if mean_distance <= np.finfo(np.float64).eps:
        raise ValueError("Correspondence points must not all be identical")

    scale = np.sqrt(2.0) / mean_distance
    transform = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0],
    ])
    homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    normalized = (transform @ homogeneous.T).T[:, :2]
    return normalized, transform


def computeH(im1_pts: np.ndarray, im2_pts: np.ndarray) -> np.ndarray:
    """Estimate H such that im2_pts ~= H @ im1_pts in homogeneous coordinates."""
    source = np.asarray(im1_pts, dtype=np.float64)
    destination = np.asarray(im2_pts, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("im1_pts must have shape (n, 2)")
    if destination.ndim != 2 or destination.shape[1] != 2:
        raise ValueError("im2_pts must have shape (n, 2)")
    if source.shape != destination.shape or source.shape[0] < 4:
        raise ValueError("Provide at least four paired points with matching shapes")
    if not np.isfinite(source).all() or not np.isfinite(destination).all():
        raise ValueError("Correspondence points must be finite")

    source_normalized, source_transform = _normalize_points(source)
    destination_normalized, destination_transform = _normalize_points(destination)
    x, y = source_normalized.T
    u, v = destination_normalized.T

    system = np.zeros((2 * source.shape[0], 8), dtype=np.float64)
    values = np.empty(2 * source.shape[0], dtype=np.float64)
    system[0::2] = np.column_stack((
        x, y, np.ones_like(x), np.zeros_like(x), np.zeros_like(x),
        np.zeros_like(x), -u * x, -u * y,
    ))
    system[1::2] = np.column_stack((
        np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), x, y,
        np.ones_like(x), -v * x, -v * y,
    ))
    values[0::2] = u
    values[1::2] = v

    parameters, _, rank, _ = np.linalg.lstsq(system, values, rcond=None)
    if rank < 8:
        raise ValueError("Correspondences do not determine a unique homography")

    normalized_homography = np.append(parameters, 1.0).reshape(3, 3)
    homography = np.linalg.inv(destination_transform) @ normalized_homography @ source_transform
    if abs(homography[2, 2]) > np.finfo(np.float64).eps:
        homography /= homography[2, 2]
    else:
        homography /= np.linalg.norm(homography)
    return homography


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Apply a homography to an (n, 2) array of (x, y) points."""
    points = np.asarray(points, dtype=np.float64)
    homography = np.asarray(homography, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if homography.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3)")

    homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    projected = (homography @ homogeneous.T).T
    denominator = projected[:, 2]
    if np.any(np.isclose(denominator, 0.0)):
        raise ValueError("Homography maps one or more points to infinity")
    return projected[:, :2] / denominator[:, None]