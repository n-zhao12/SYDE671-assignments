from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


DX = np.array([[1.0, 0.0, -1.0]])
DY = DX.T.copy()


def read_grayscale(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image.astype(np.float64)


def save_image(path: Path, image: np.ndarray, title: str, cmap: str = "gray") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_image_grid(path: Path, images: list[tuple[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 4))
    if len(images) == 1:
        axes = [axes]
    for axis, (title, image) in zip(axes, images):
        axis.imshow(image, cmap="gray")
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    low, high = np.min(image), np.max(image)
    if high == low:
        return np.zeros_like(image, dtype=np.float64)
    return (image - low) / (high - low)


def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)