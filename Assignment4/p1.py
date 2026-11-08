import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from epipolar_utils import *

'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using 
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the linear least squares eight
point algorithm works
'''
def lls_eight_point_alg(points1, points2):
    if points1.shape != points2.shape or points1.ndim != 2 or points1.shape[1] != 3:
        raise ValueError('points1 and points2 must have the same Nx3 shape')
    if points1.shape[0] < 8:
        raise ValueError('At least eight point correspondences are required')

    x1, y1 = points1[:, 0] / points1[:, 2], points1[:, 1] / points1[:, 2]
    x2, y2 = points2[:, 0] / points2[:, 2], points2[:, 1] / points2[:, 2]
    design = np.column_stack((x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, np.ones_like(x1)))
    _, _, right = np.linalg.svd(design)
    F = right[-1].reshape(3, 3)

    left, singular_values, right = np.linalg.svd(F)
    singular_values[-1] = 0
    F = left @ np.diag(singular_values) @ right
    return F / np.linalg.norm(F)

'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the normalized eight
point algorithm works
'''
def normalized_eight_point_alg(points1, points2):
    if points1.shape != points2.shape or points1.ndim != 2 or points1.shape[1] != 3:
        raise ValueError('points1 and points2 must have the same Nx3 shape')
    if points1.shape[0] < 8:
        raise ValueError('At least eight point correspondences are required')

    def normalize(points):
        xy = points[:, :2] / points[:, 2, None]
        center = xy.mean(axis=0)
        mean_distance = np.linalg.norm(xy - center, axis=1).mean()
        if mean_distance == 0:
            raise ValueError('Point correspondences cannot all be identical')
        scale = np.sqrt(2) / mean_distance
        transform = np.array([[scale, 0, -scale * center[0]],
                              [0, scale, -scale * center[1]],
                              [0, 0, 1]])
        normalized = (transform @ points.T).T
        normalized /= normalized[:, 2, None]
        return normalized, transform

    normalized1, T1 = normalize(points1)
    normalized2, T2 = normalize(points2)
    F = lls_eight_point_alg(normalized1, normalized2)
    F = T2.T @ F @ T1
    left, singular_values, right = np.linalg.svd(F)
    singular_values[-1] = 0
    F = left @ np.diag(singular_values) @ right
    return F / np.linalg.norm(F)

'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image 
    im2 - a HxW(xC) matrix that contains pixel values from the second image 
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):

    def plot_epipolar_lines_on_image(points1, points2, im, F):
        im_height = im.shape[0]
        im_width = im.shape[1]
        lines = F.T.dot(points2.T)
        plt.imshow(im, cmap='gray')
        for line in lines.T:
            a,b,c = line
            xs = [1, im.shape[1]-1]
            ys = [(-c-a*x)/b for x in xs]
            plt.plot(xs, ys, 'r')
        for i in range(points1.shape[0]):
            x,y,_ = points1[i]
            plt.plot(x, y, '*b')
        plt.axis([0, im_width, im_height, 0])

    # We change the figsize because matplotlib has weird behavior when 
    # plotting images of different sizes next to each other. This
    # fix should be changed to something more robust.
    new_figsize = (8 * (float(max(im1.shape[1], im2.shape[1])) / min(im1.shape[1], im2.shape[1]))**2 , 6)
    fig = plt.figure(figsize=new_figsize)
    plt.subplot(121)
    plot_epipolar_lines_on_image(points1, points2, im1, F)
    plt.axis('off')
    plt.subplot(122)
    plot_epipolar_lines_on_image(points2, points1, im2, F.T)
    plt.axis('off')

'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
points to their corresponding epipolar lines. Compute just the average distance
from points1 to their corresponding epipolar lines (which you get from points2).
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    lines = (F.T @ points2.T).T
    points = points1[:, :2] / points1[:, 2, None]
    line_norms = np.linalg.norm(lines[:, :2], axis=1)
    if np.any(line_norms == 0):
        raise ValueError('Cannot compute distance to a degenerate epipolar line')
    distances = np.abs(np.sum(lines[:, :2] * points, axis=1) + lines[:, 2]) / line_norms
    return distances.mean()

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print('-'*80)
        print("Set:", im_set)
        print('-'*80)

        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        F_lls = lls_eight_point_alg(points1, points2)
        print("Fundamental Matrix from LLS  8-point algorithm:\n", F_lls)
        print("Distance to lines in image 1 for LLS:", \
            compute_distance_to_epipolar_lines(points1, points2, F_lls))
        print("Distance to lines in image 2 for LLS:", \
            compute_distance_to_epipolar_lines(points2, points1, F_lls.T))

        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points2[i].dot(F_normalized.dot(points1[i])) 
            for i in range(points1.shape[0])]
        print("p'^T F p =", np.abs(pFp).max())
        print("Fundamental Matrix from normalized 8-point algorithm:\n", \
            F_normalized)
        print("Distance to lines in image 1 for normalized:", \
            compute_distance_to_epipolar_lines(points1, points2, F_normalized))
        print("Distance to lines in image 2 for normalized:", \
            compute_distance_to_epipolar_lines(points2, points1, F_normalized.T))

        # Plotting the epipolar lines
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)

        plt.show()
