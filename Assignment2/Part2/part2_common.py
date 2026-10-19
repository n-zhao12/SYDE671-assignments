from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "part2_outputs"
MAX_IMAGE_SIDE = 768


def read_rgb(path: str | Path, max_side: int = MAX_IMAGE_SIDE) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image


def save_grid(path: Path, panels: list[tuple[str, np.ndarray]], columns: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = (len(panels) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.6 * rows), squeeze=False)
    for axis, (title, image) in zip(axes.flat, panels):
        if image.ndim == 2:
            axis.imshow(np.clip(image, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
        else:
            axis.imshow(np.clip(image, 0.0, 1.0))
        axis.set_title(title)
        axis.axis("off")
    for axis in axes.flat[len(panels):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REFLECT101)


def signed_display(image: np.ndarray) -> np.ndarray:
    scale = float(np.percentile(np.abs(image), 99.5))
    if scale == 0.0:
        return np.full_like(image, 0.5, dtype=np.float32)
    return np.clip(0.5 + image / (2.0 * scale), 0.0, 1.0)


def fit_to_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = shape
    height, width = image.shape[:2]
    scale = max(target_width / width, target_height / height)
    resized = cv2.resize(
        image,
        (max(target_width, round(width * scale)), max(target_height, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    y0 = (resized.shape[0] - target_height) // 2
    x0 = (resized.shape[1] - target_width) // 2
    return resized[y0:y0 + target_height, x0:x0 + target_width]