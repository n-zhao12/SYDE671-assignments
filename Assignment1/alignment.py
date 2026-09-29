import cv2
import numpy as np


def ncc(img1, img2):

    img1 = img1 - np.mean(img1)
    img2 = img2 - np.mean(img2)

    denom = (
        np.linalg.norm(img1) *
        np.linalg.norm(img2)
    )

    if denom == 0:
        return -1

    return np.sum(img1 * img2) / denom


def gradient_image(img):
    """Return Sobel gradient magnitude for an image."""
    img = img.astype(np.float32, copy=False)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def gradient_ncc(img1, img2):
    """Compute NCC between Sobel gradient magnitudes."""
    return ncc(gradient_image(img1), gradient_image(img2))


def texture_weights(img, strength=4.0):
    """Give textured reference pixels more influence during NCC scoring."""
    magnitude = gradient_image(img)
    scale = np.percentile(magnitude, 95)
    if scale <= 0:
        return np.ones_like(magnitude, dtype=np.float32)

    texture = np.clip(magnitude / scale, 0.0, 1.0)
    return 1.0 + strength * texture**2


def weighted_ncc(img1, img2, weights):
    """NCC with a fixed spatial weight map, centered by weighted means."""
    weights = weights.astype(np.float32, copy=False)
    weight_sum = np.sum(weights)
    if weight_sum <= 0:
        return -1

    img1 = img1.astype(np.float32, copy=False)
    img2 = img2.astype(np.float32, copy=False)
    mean1 = np.sum(weights * img1) / weight_sum
    mean2 = np.sum(weights * img2) / weight_sum
    centered1 = img1 - mean1
    centered2 = img2 - mean2
    denom = np.sqrt(
        np.sum(weights * centered1**2) *
        np.sum(weights * centered2**2)
    )
    if denom == 0:
        return -1
    return np.sum(weights * centered1 * centered2) / denom


def l2_distance(img1, img2):

    return np.sqrt(
        np.sum((img1 - img2) ** 2)
    )


def aligned_overlap(
    reference,
    target,
    dx,
    dy,
    edge_fraction=0.05,
    weights=None
):
    """Return matching regions without the wrapped pixels from np.roll."""
    height, width = reference.shape

    y_start = max(0, dy)
    y_end = min(height, height + dy)
    x_start = max(0, dx)
    x_end = min(width, width + dx)

    # Ignore a proportional edge margin so plate frames matter less at every scale.
    margin_y = int((y_end - y_start) * edge_fraction)
    margin_x = int((x_end - x_start) * edge_fraction)
    y_start += margin_y
    y_end -= margin_y
    x_start += margin_x
    x_end -= margin_x

    reference_region = reference[y_start:y_end, x_start:x_end]
    target_region = target[
        y_start - dy:y_end - dy,
        x_start - dx:x_end - dx
    ]
    if weights is not None:
        weight_region = weights[y_start:y_end, x_start:x_end]
        return reference_region, target_region, weight_region
    return reference_region, target_region


def align_channel(reference, target, search_range=15, metric="ncc", edge_fraction=0.05):

    if metric == "gradient_ncc":
        reference_features = gradient_image(reference)
        target_features = gradient_image(target)
    else:
        reference_features = reference
        target_features = target
    weights = texture_weights(reference) if metric == "weighted_ncc" else None

    best_shift = (0, 0)

    if metric in ("ncc", "gradient_ncc", "weighted_ncc"):
        best_score = -np.inf
    else:
        best_score = np.inf

    for dx in range(
        -search_range,
        search_range + 1
    ):

        for dy in range(
            -search_range,
            search_range + 1
        ):

            overlap = aligned_overlap(
                reference_features,
                target_features,
                dx,
                dy,
                edge_fraction=edge_fraction,
                weights=weights
            )
            if weights is not None:
                ref_crop, shifted_crop, weight_crop = overlap
            else:
                ref_crop, shifted_crop = overlap

            if metric == "weighted_ncc":
                score = weighted_ncc(ref_crop, shifted_crop, weight_crop)
                if score > best_score:
                    best_score = score
                    best_shift = (dx, dy)
            elif metric in ("ncc", "gradient_ncc"):

                score = ncc(
                    ref_crop,
                    shifted_crop
                )

                if score > best_score:
                    best_score = score
                    best_shift = (dx, dy)

            else:

                score = l2_distance(
                    ref_crop,
                    shifted_crop
                )

                if score < best_score:
                    best_score = score
                    best_shift = (dx, dy)

    return best_shift
