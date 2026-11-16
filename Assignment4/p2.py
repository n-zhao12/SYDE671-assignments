import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from p1 import *
from epipolar_utils import *

'''
COMPUTE_EPIPOLE computes the epipole e in homogenous coordinates
given the fundamental matrix
Arguments:
    F - the Fundamental matrix solved for with normalized_eight_point_alg(points1, points2)

Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(F):
    left, _, _ = np.linalg.svd(F)
    epipole = left[:, -1]
    if abs(epipole[2]) > 1e-12:
        epipole = epipole / epipole[2]
    else:
        epipole = epipole / np.linalg.norm(epipole[:2])
    return epipole

def compute_H(e, im):
    height, width = im.shape[:2]
    T = np.array([[1, 0, -width / 2],
                  [0, 1, -height / 2],
                  [0, 0, 1]], dtype=float)
    centered_epipole = T @ e
    radius = np.linalg.norm(centered_epipole[:2])
    if radius < 1e-12:
        raise ValueError('The epipole cannot coincide with the image center')

    cosine = centered_epipole[0] / radius
    sine = centered_epipole[1] / radius
    alpha = 1 if centered_epipole[0] >= 0 else -1
    R = np.array([[alpha * cosine, alpha * sine, 0],
                  [-alpha * sine, alpha * cosine, 0],
                  [0, 0, 1]], dtype=float)
    rotated_epipole = R @ centered_epipole
    G = np.eye(3)
    if abs(rotated_epipole[2]) > 1e-12:
        G[2, 0] = -rotated_epipole[2] / rotated_epipole[0]
    return np.linalg.inv(T) @ G @ R @ T

def compute_matching_homographies(e2, F, im2, points1, points2):
    H2 = compute_H(e2, im2)
    skew_e2 = np.array([[0, -e2[2], e2[1]],
                        [e2[2], 0, -e2[0]],
                        [-e2[1], e2[0], 0]])
    base_mapping = skew_e2 @ F
    target_points_h = (H2 @ points2.T).T
    target_points = target_points_h / target_points_h[:, 2, None]

    v = np.ones(3)
    M = base_mapping + np.outer(e2, v)

    rectified1_h = (H2 @ M @ points1.T).T
    rectified1 = rectified1_h / rectified1_h[:, 2, None]
    affine_design = np.column_stack((rectified1[:, 0], rectified1[:, 1],
                                     np.ones(points1.shape[0])))
    affine_x, _, _, _ = np.linalg.lstsq(affine_design, target_points[:, 0], rcond=None)
    H_affine = np.array([[affine_x[0], affine_x[1], affine_x[2]],
                         [0, 1, 0],
                         [0, 0, 1]])
    H1 = H_affine @ H2 @ M
    return H1, H2

if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    # F is such that such that (points2)^T * F * points1 = 0, so e1 is e' and e2 is e
    e1 = compute_epipole(F.T)
    e2 = compute_epipole(F)
    print("e1", e1)
    print("e2", e2)

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print('')

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()
