import argparse
from pathlib import Path

from utils import (
    load_channels,
    shift_image,
    create_color_image,
    save_image,
    auto_contrast
)

from pyramid import pyramid_align
from alignment import align_channel


def compose_aligned_channels(B, G, R, g_shift, r_shift):
    G_aligned = shift_image(G, g_shift[0], g_shift[1])
    R_aligned = shift_image(R, r_shift[0], r_shift[1])
    color_img = create_color_image(B, G_aligned, R_aligned)

    # Keep only pixels that came from all three original channels.
    height, width = color_img.shape[:2]
    shifts = (g_shift, r_shift)
    top = max(0, *(dy for _, dy in shifts))
    bottom = min(height, *(height + dy for _, dy in shifts))
    left = max(0, *(dx for dx, _ in shifts))
    right = min(width, *(width + dx for dx, _ in shifts))
    return auto_contrast(color_img[top:bottom, left:right])


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

    return compose_aligned_channels(B, G, R, g_shift, r_shift)


def process_image_single_scale(
    image_path,
    metric="ncc",
    search_range=15,
    edge_fraction=0.05
):
    """Align both channels directly at the input resolution."""
    B, G, R = load_channels(image_path)
    g_shift = align_channel(
        B, G, search_range=search_range,
        metric=metric, edge_fraction=edge_fraction
    )
    r_shift = align_channel(
        B, R, search_range=search_range,
        metric=metric, edge_fraction=edge_fraction
    )
    print(f"Single-scale G shift ({metric}): {g_shift}")
    print(f"Single-scale R shift ({metric}): {r_shift}")
    return compose_aligned_channels(B, G, R, g_shift, r_shift)


def run_single_scale_examples():
    image_names = ("00056v.jpg", "00125v.jpg")
    search_range = 15

    output_dir = Path("single_scale_ncc_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_name in image_names:
        image_path = Path("images") / image_name
        result = process_image_single_scale(
            str(image_path), metric="ncc",
            search_range=search_range, edge_fraction=0.05
        )
        output_path = output_dir / f"{image_path.stem}_result.jpg"
        save_image(str(output_path), result)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single-scale",
        action="store_true",
        help="Run single-scale NCC alignment on two low-resolution JPGs."
    )
    args = parser.parse_args()

    if args.single_scale:
        run_single_scale_examples()
    else:
        alignment_metric = "ncc"
        edge_fraction = 0.05
        input_dir = Path("self_picked_images")
        output_folders = {
            "ncc": "raw_ncc_5pct_results",
            "gradient_ncc": "gradient_ncc_results",
            "weighted_ncc": "texture_weighted_ncc_results",
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
