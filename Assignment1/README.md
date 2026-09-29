# SYDE671-assignment 1 - Prokudin-Gorskii Photo Colorization

This project reconstructs color images from Prokudin-Gorskii glass plate scans. Each input is a grayscale image containing three vertically stacked exposures, ordered blue, green, and red from top to bottom. The program separates the exposures, aligns green and red to blue using translations, and combines them into a color image.

## Current pipeline

The current `main.py` configuration processes the JPG files in `self_picked_images/` using **texture-weighted NCC** with a 5% edge exclusion. Results are written to `texture_weighted_ncc_results/`.

For each input image, the program:

1. Reads the grayscale plate and divides it into three equal-height blue, green, and red channels.
2. Aligns green and red independently to blue with a coarse-to-fine image pyramid.
3. Scores each translation using texture-weighted normalized cross-correlation (NCC), comparing only valid overlapping pixels and ignoring 5% around the overlap edges.
4. Shifts the channels, crops to the area shared by all three, combines them in OpenCV's BGR order, and applies global automatic contrast.
5. Prints the `(dx, dy)` displacement applied to green and red and saves a JPG result.

The pyramid downsamples by a factor of two with area interpolation. It searches a range of ±15 pixels at its coarsest level, then refines the estimate within ±5 pixels at each finer level.

### Texture-weighted NCC

The weight map is computed from the blue reference channel's Sobel gradient magnitude. Gradient values are normalized using their 95th percentile and clipped to `[0, 1]`. The pixel weight is `1 + 4t²`, where `t` is the normalized gradient magnitude. This gives detailed regions more influence in the NCC score while still including smoother regions.

## Methods implemented

- **Raw NCC**: mean-centered normalized cross-correlation of pixel intensities.
- **Texture-weighted NCC**: raw-intensity NCC with greater weight on detailed regions; this is the current default.
- **Gradient NCC**: NCC applied to Sobel gradient magnitudes.
- **L2 distance**: Euclidean pixel difference, minimized during alignment.
- **Single-scale alignment**: `align_channel()` in `alignment.py` exhaustively searches a user-specified translation window. It is implemented, but the current `main.py` workflow uses the pyramid and a standalone single-scale results run has not yet been done.

The metric can be changed in the `alignment_metric` setting near the bottom of `main.py`. The input folder and output folder are set there as well.

## Running the program

Install the Python dependencies if needed:

```bash
python -m pip install numpy opencv-python
```

Place stacked JPG images in `self_picked_images/`, then run from the project directory:

```bash
python main.py
```

To process the provided collection images instead, change `input_dir` in `main.py` to `Path("images")`. The output folder is selected from the metric-to-folder mapping in that file.

## Current self-picked image offsets

These are the offsets printed by the current texture-weighted NCC pipeline. They are the translations applied to green and red to align each channel to blue.

| Image | Green `(dx, dy)` | Red `(dx, dy)` |
|---|---:|---:|
| `00279v.jpg` | `(1, 3)` | `(2, 12)` |
| `00470v.jpg` | `(-2, 6)` | `(-5, 12)` |
| `02180v.jpg` | `(-2, 1)` | `(-3, 1)` |

## Experiment outputs

Separate folders contain results from the methods compared during development:

- `raw_ncc_5pct_results/` — raw NCC with overlap-aware scoring and 5% edge exclusion.
- `gradient_ncc_results/` — gradient-magnitude NCC.
- `texture_weighted_ncc_results/` — texture-weighted NCC.
- `raw_ncc_5pct_wide_refine_results/` — a wider final-resolution search comparison for `01007a`; it selected the same shifts as the standard raw-NCC run.

Texture-weighted NCC has looked best so far on the images reviewed, but it does not improve every image. Some color fringing remains, especially around high-contrast details and plate borders. The current model estimates translation only; it does not correct scale, rotation, local distortion, or automatically remove the original plate frame.

## Project files

- `main.py` — input selection, channel alignment, output cropping, and batch processing.
- `utils.py` — channel loading, translation, color image assembly, saving, and contrast adjustment.
- `alignment.py` — NCC, weighted NCC, gradient NCC, L2 distance, overlap handling, and single-scale search.
- `pyramid.py` — downsampling and coarse-to-fine alignment.
