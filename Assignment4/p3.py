import numpy as np
from scipy.optimize import least_squares
from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.gridspec as gridspec
from epipolar_utils import *

'''
FACTORIZATION_METHOD The Tomasi and Kanade Factorization Method to determine
the 3D structure of the scene and the motion of the cameras.
Arguments:
    points_im1 - N points in the first image that match with points_im2
    points_im2 - N points in the second image that match with points_im1

    Both points_im1 and points_im2 are from the get_data_from_txt_file() method
Returns:
    structure - the structure matrix
    motion - the motion matrix
'''
def factorization_method(points_im1, points_im2):
    if points_im1.shape != points_im2.shape or points_im1.ndim != 2 or points_im1.shape[1] != 3:
        raise ValueError('Both point arrays must have the same Nx3 shape')
    if points_im1.shape[0] < 3:
        raise ValueError('At least three point correspondences are required')

    measurement = np.vstack((points_im1[:, :2].T, points_im2[:, :2].T)).astype(float)
    measurement -= measurement.mean(axis=1, keepdims=True)
    U, singular_values, Vt = np.linalg.svd(measurement, full_matrices=False)
    root_singular_values = np.diag(np.sqrt(singular_values[:3]))
    motion = U[:, :3] @ root_singular_values
    structure = root_singular_values @ Vt[:3, :]
    raw_motion = motion.copy()
    raw_structure = structure.copy()

    def quadratic_coefficients(first, second):
        return np.array([
            first[0] * second[0],
            first[0] * second[1] + first[1] * second[0],
            first[0] * second[2] + first[2] * second[0],
            first[1] * second[1],
            first[1] * second[2] + first[2] * second[1],
            first[2] * second[2]])

    constraints = []
    for camera in range(2):
        first_row, second_row = motion[2*camera], motion[2*camera + 1]
        constraints.append(quadratic_coefficients(first_row, second_row))
        constraints.append(quadratic_coefficients(first_row, first_row) -
                           quadratic_coefficients(second_row, second_row))
    constraints.append(quadratic_coefficients(motion[0], motion[0]) -
                       quadratic_coefficients(motion[2], motion[2]))

    constraints = np.asarray(constraints)
    constraints /= np.linalg.norm(constraints, axis=1, keepdims=True)

    isotropic_metric = np.array([1/3, 0, 0, 1/3, 0, 1/3])

    def metric_residual(parameters):
        L = np.array([[np.exp(parameters[0]), 0, 0],
                      [parameters[1], np.exp(parameters[2]), 0],
                      [parameters[3], parameters[4], np.exp(parameters[5])]])
        metric = L @ L.T
        metric /= np.trace(metric)
        q = np.array([metric[0, 0], metric[0, 1], metric[0, 2],
                      metric[1, 1], metric[1, 2], metric[2, 2]])
        return np.concatenate((constraints @ q,
                       0.03 * (q - isotropic_metric)))

    metric_fit = least_squares(metric_residual, np.zeros(6))
    parameters = metric_fit.x
    L = np.array([[np.exp(parameters[0]), 0, 0],
                  [parameters[1], np.exp(parameters[2]), 0],
                  [parameters[3], parameters[4], np.exp(parameters[5])]])
    metric = L @ L.T
    metric /= np.trace(metric) / 3
    metric_eigenvalues = np.linalg.eigvalsh(metric)
    constraint_error = np.linalg.norm(metric_residual(parameters)[:len(constraints)])
    if metric_eigenvalues[0] <= 0 or metric_eigenvalues[-1] / metric_eigenvalues[0] > 100 or constraint_error > 0.05:
        return raw_structure, raw_motion

    upgrade = np.linalg.cholesky(metric)
    motion = motion @ upgrade
    structure = np.linalg.solve(upgrade, structure)
    return structure, motion

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set1_subset']:
        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points_im1 = get_data_from_txt_file(im_set + '/pt_2D_1.txt')
        points_im2 = get_data_from_txt_file(im_set + '/pt_2D_2.txt')
        points_3d = get_data_from_txt_file(im_set + '/pt_3D.txt')
        assert (points_im1.shape == points_im2.shape)

        # Run the Factorization Method
        structure, motion = factorization_method(points_im1, points_im2)

        # Plot the structure
        fig = plt.figure()
        ax = fig.add_subplot(121, projection = '3d')
        scatter_3D_axis_equal(structure[0,:], structure[1,:], structure[2,:], ax)
        ax.set_title('Factorization Method')
        ax = fig.add_subplot(122, projection = '3d')
        scatter_3D_axis_equal(points_3d[:,0], points_3d[:,1], points_3d[:,2], ax)
        ax.set_title('Ground Truth')

        plt.show()
