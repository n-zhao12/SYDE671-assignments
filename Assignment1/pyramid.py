import cv2
import numpy as np

from alignment import (
    aligned_overlap,
    gradient_image,
    l2_distance,
    ncc,
    texture_weights,
    weighted_ncc,
)


def downsample(img):

    return cv2.resize(
        img,
        (img.shape[1] // 2,
         img.shape[0] // 2),
        interpolation=cv2.INTER_AREA
    )


def pyramid_align(
    reference,
    target,
    level=4,
    metric="ncc",
    edge_fraction=0.05
):

    if (
        level == 0 or
        min(reference.shape) < 400
    ):

        from alignment import align_channel

        return align_channel(
            reference,
            target,
            search_range=15,
            metric=metric,
            edge_fraction=edge_fraction
        )

    small_ref = downsample(
        reference
    )

    small_target = downsample(
        target
    )

    dx, dy = pyramid_align(
        small_ref,
        small_target,
        level - 1,
        metric=metric,
        edge_fraction=edge_fraction
    )

    dx *= 2
    dy *= 2

    if metric == "gradient_ncc":
        reference_features = gradient_image(reference)
        target_features = gradient_image(target)
    else:
        reference_features = reference
        target_features = target
    weights = texture_weights(reference) if metric == "weighted_ncc" else None

    best_score = np.inf if metric == "l2" else -np.inf
    best_shift = (dx, dy)

    for x in range(dx - 5, dx + 6):
        for y in range(dy - 5, dy + 6):

            overlap = aligned_overlap(
                reference_features,
                target_features,
                x,
                y,
                edge_fraction=edge_fraction,
                weights=weights
            )
            if weights is not None:
                ref_crop, shifted_crop, weight_crop = overlap
            else:
                ref_crop, shifted_crop = overlap

            if metric == "l2":
                score = l2_distance(ref_crop, shifted_crop)
            elif metric == "weighted_ncc":
                score = weighted_ncc(ref_crop, shifted_crop, weight_crop)
            else:
                score = ncc(ref_crop, shifted_crop)

            is_better = score < best_score if metric == "l2" else score > best_score
            if is_better:

                best_score = score
                best_shift = (x, y)

    return best_shift
