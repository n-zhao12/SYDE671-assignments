from pathlib import Path

from utils import (
    load_channels,
    shift_image,
    create_color_image,
    save_image,
    auto_contrast
)

from pyramid import pyramid_align


def process_image(image_path, metric="ncc", edge_fraction=0.05):

    B, G, R = load_channels(
        image_path
    )

    g_shift = pyramid_align(
        B,
        G,
        metric=metric,
        edge_fraction=edge_fraction
    )

    r_shift = pyramid_align(
        B,
        R,
        metric=metric,
        edge_fraction=edge_fraction
    )

    print(
        f"G shift ({metric}): {g_shift}"
    )

    print(
        f"R shift ({metric}): {r_shift}"
    )

    G_aligned = shift_image(
        G,
        g_shift[0],
        g_shift[1]
    )

    R_aligned = shift_image(
        R,
        r_shift[0],
        r_shift[1]
    )

    color_img = create_color_image(
        B,
        G_aligned,
        R_aligned
    )

    # Keep only pixels that came from all three original channels.
    height, width = color_img.shape[:2]
    shifts = (g_shift, r_shift)
    top = max(0, *(dy for _, dy in shifts))
    bottom = min(height, *(height + dy for _, dy in shifts))
    left = max(0, *(dx for dx, _ in shifts))
    right = min(width, *(width + dx for dx, _ in shifts))
    color_img = color_img[top:bottom, left:right]

    color_img = auto_contrast(
        color_img
    )

    return color_img


if __name__ == "__main__":
    alignment_metric = "weighted_ncc"
    edge_fraction = 0.05
    input_dir = Path("self_picked_images")
    output_folders = {
        "ncc": "raw_ncc_5pct_results",
        "gradient_ncc": "gradient_ncc_results",
        "weighted_ncc": "texture_weighted_ncc_results",
        "l2": "l2_results",
    }
    output_dir = Path(output_folders[alignment_metric])
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found in {input_dir.resolve()}")

    for image_path in image_paths:
        output_path = output_dir / f"{image_path.stem}_result.jpg"

        print(f"Processing {image_path}...")
        result = process_image(
            str(image_path),
            metric=alignment_metric,
            edge_fraction=edge_fraction
        )
        save_image(str(output_path), result)
        print(f"Saved {output_path}")
