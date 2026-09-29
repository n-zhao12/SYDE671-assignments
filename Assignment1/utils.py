import cv2
import numpy as np


def load_channels(filename):

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)
    h = img.shape[0] // 3

    B = img[:h]
    G = img[h:2*h]
    R = img[2*h:3*h]

    return B, G, R


def shift_image(img, dx, dy):
    return np.roll(img, shift=(dy, dx), axis=(0, 1))


def create_color_image(B, G, R):

    # OpenCV uses BGR channel order for images written with cv2.imwrite.
    color = np.dstack([B, G, R])
    color = np.clip(color, 0, 255)

    return color.astype(np.uint8)


def save_image(filename, image):
    cv2.imwrite(filename, image)


def auto_contrast(img):

    img = img.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()

    img *= 255

    return img.astype(np.uint8)
