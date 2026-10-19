"""Run the Assignment 2 Part 1 subparts from one entry point."""

import argparse
from pathlib import Path

from part1_1_convolution import benchmark_convolutions, test_convolutions
from part1_2_finite_difference import finite_difference, run_finite_difference_section
from part1_3_dog import run_dog_section
from part1_orientation import run_orientation_section
from part1_utils import read_grayscale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfie", type=str, default="selfie.jpg",
                        help="Path to your own portrait image")
    parser.add_argument("--cameraman", type=str, default="cameraman.png",
                        help="Path to the cameraman image")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Optional gradient-magnitude threshold")
    parser.add_argument("--sigma", type=float, default=1.5,
                        help="Gaussian sigma for Part 1.3")
    parser.add_argument("--gaussian-size", type=int, default=9,
                        help="Odd Gaussian kernel size for Part 1.3")
    parser.add_argument("--part", choices=("all", "1.1", "1.2", "1.3", "bells"),
                        default="all", help="Run one subpart or all of Part 1")
    args = parser.parse_args()

    output_dir = Path("part1_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.part in ("all", "1.1"):
        test_convolutions()
        if args.selfie:
            benchmark_convolutions(args.selfie, output_dir)
        else:
            print("No selfie supplied; skipping Part 1.1 selfie output.")

    if args.part in ("all", "1.2", "1.3", "bells"):
        cameraman = read_grayscale(args.cameraman)
        finite_results = None
        if args.part in ("all", "1.2"):
            finite_results = run_finite_difference_section(
                cameraman, output_dir, threshold=args.threshold
            )
        if args.part in ("all", "1.3"):
            run_dog_section(
                cameraman,
                output_dir,
                gaussian_size=args.gaussian_size,
                sigma=args.sigma,
                threshold=args.threshold,
            )
        if args.part in ("all", "bells"):
            if finite_results is None:
                gx, gy = finite_difference(cameraman)
            else:
                gx, gy = finite_results["gx"], finite_results["gy"]
            run_orientation_section(gx, gy, output_dir)

    print("Finished. Results are in:", output_dir.resolve())


if __name__ == "__main__":
    main()
