# SYDE 671 Assignment 1, Part 2 — Prokudin-Gorskii Photo Colorization

This project reconstructs color images from Prokudin-Gorskii glass-plate scans. Each input is a grayscale image with three vertically stacked exposures, ordered blue, green, and red from top to bottom. The program separates the exposures, aligns green and red to blue, and combines them into a color image.

## Selected method: normalized cross-correlation

The selected alignment metric is raw normalized cross-correlation (NCC). Each channel is mean-centered and normalized before comparing a candidate shift. The separate color-filter exposures can have different brightness levels, so NCC is a reasonable choice: mean-centering and normalization make the score less dependent on absolute intensity and more dependent on the shared scene structure. The best-scoring shift is selected.

## Pipeline

The current `main.py` workflow processes JPG images from `images/` using pyramid alignment, raw NCC, and a 5% edge exclusion. Results are saved in `raw_ncc_5pct_results/`.

For each plate, the program:

1. Splits the grayscale image into equal-height blue, green, and red sections.
2. Aligns green and red independently to blue using a coarse-to-fine image pyramid.
3. Scores candidate shifts using NCC over valid overlapping regions, excluding a 5% margin at the overlap edges.
4. Applies the translations, crops to the region shared by all three channels, combines the channels, and rescales contrast.
5. Prints the `(dx, dy)` displacement vectors applied to green and red, then saves a JPG result.

The pyramid downsamples by two using area interpolation. It searches ±15 pixels at the coarsest level, then refines the estimate within ±5 pixels at each finer level. The edge exclusion affects alignment scoring; it is not automatic detection/removal of the plate frame.

## Single-scale alignment

`align_channel()` in `alignment.py` exhaustively searches a user-specified translation window. Run the single-scale NCC examples with:

```bash
python main.py --single-scale
```

This processes `00056v.jpg` and `00125v.jpg` from `images/` with a ±15-pixel search and writes results to `single_scale_ncc_results/`.

## Additional experiments

The code also includes two optional NCC variants:

- **Gradient NCC** compares Sobel gradient magnitudes. It was tested, but it was not selected because results were less reliable on the reviewed images.
- **Texture-weighted NCC** uses Sobel magnitude from the blue reference to give detailed regions more influence. Results were visually similar to raw NCC in the examples reviewed, so raw NCC with edge exclusion remains the selected method.

The 5% edge exclusion was compared with raw NCC without the exclusion and gave better alignments on the reviewed results. A wider final-resolution refinement trial for `01007a` selected the same shifts as the standard raw-NCC run.

## Run the full provided image set

Install dependencies if needed:

```bash
python -m pip install numpy opencv-python
```

From this project directory, run:

```bash
python main.py
```

The default configuration processes JPGs in `images/` and saves results in `raw_ncc_5pct_results/`.

## Result folders

- `NCC_results/` — raw NCC without edge exclusion.
- `raw_ncc_5pct_results/` — selected raw NCC method with 5% edge exclusion.
- `single_scale_ncc_results/` — exhaustive single-scale NCC examples.
- `gradient_ncc_results/` — gradient-magnitude NCC experiment.
- `texture_weighted_ncc_results/` — texture-weighted NCC experiment.
- `raw_ncc_5pct_wide_refine_results/` — wider refinement comparison for `01007a`.

## Project files

- `main.py` — pipeline, image selection, batch processing, and single-scale examples.
- `utils.py` — channel loading, shifting, image assembly, saving, and contrast adjustment.
- `alignment.py` — NCC variants, overlap handling, and exhaustive single-scale search.
- `pyramid.py` — downsampling and coarse-to-fine alignment.