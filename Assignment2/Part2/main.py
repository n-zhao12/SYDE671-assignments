"""Run Assignment 2 Part 2.1 (sharpening) and 2.2 (hybrid images)."""

import argparse
from pathlib import Path

from part2_1_sharpening import run_sharpening
from part2_2_hybrid_images import (
	DEREK_EYE_POINTS,
	NUTMEG_EYE_POINTS,
	run_hybrid,
)
from part2_common import OUTPUT_DIR, ROOT

CUSTOM_HYBRID_PAIRS = (
	("harry_voldemort", ROOT / "hybrid" / "pair1_Voldemort.jpg", ROOT / "hybrid" / "pair1_Harry.jpg"),
	("lion_tiger", ROOT / "hybrid" / "pair2_lion.jpg", ROOT / "hybrid" / "pair2_tiger.jpg"),
)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Run Assignment 2 Part 2.1 sharpening and Part 2.2 hybrids."
	)
	parser.add_argument(
		"--part",
		choices=("all", "2.1", "2.2"),
		default="all",
		help="Run both supported sections or only one",
	)
	parser.add_argument(
		"--sharpen-extra",
		type=Path,
		help="Optional sharp image for Part 2.1's blur-and-resharpen evaluation",
	)
	parser.add_argument("--low-sigma", type=float, default=8.0)
	parser.add_argument("--high-sigma", type=float, default=5.0)
	parser.add_argument(
		"--hybrid-view",
		choices=("assignment", "full", "summary"),
		default="assignment",
		help="assignment: full Derek/Nutmeg analysis and concise custom results; full: analyze every pair; summary: originals and final hybrids only",
	)
	parser.add_argument(
		"--no-align",
		action="store_true",
		help="Skip landmark calibration and only resize hybrid image pairs to match",
	)
	args = parser.parse_args()

	if args.low_sigma <= 0 or args.high_sigma <= 0:
		parser.error("Gaussian sigma values must be positive")

	if args.part in ("all", "2.1"):
		taj_image = ROOT / "images" / "taj.jpg"
		run_sharpening(taj_image, "taj", OUTPUT_DIR / "2_1_sharpening")
		if args.sharpen_extra:
			run_sharpening(args.sharpen_extra, args.sharpen_extra.stem,
						   OUTPUT_DIR / "2_1_sharpening")
		else:
			print("No --sharpen-extra supplied; skipping the second-image evaluation.")

	if args.part in ("all", "2.2"):
		hybrid_output = OUTPUT_DIR / "2_2_hybrids"
		run_hybrid(
			ROOT / "hybrid" / "DerekPicture.jpg",
			ROOT / "hybrid" / "nutmeg.jpg",
			"derek_nutmeg",
			hybrid_output,
			args.low_sigma,
			args.high_sigma,
			align=not args.no_align,
			full_analysis=args.hybrid_view in ("assignment", "full"),
			landmark_points=(DEREK_EYE_POINTS, NUTMEG_EYE_POINTS),
		)
		for label, image_a, image_b in CUSTOM_HYBRID_PAIRS:
			run_hybrid(
				image_a,
				image_b,
				label,
				hybrid_output,
				args.low_sigma,
				args.high_sigma,
				align=not args.no_align,
				full_analysis=args.hybrid_view == "full",
			)

	print("Part 2.1/2.2 outputs saved under:", OUTPUT_DIR)


if __name__ == "__main__":
	main()
