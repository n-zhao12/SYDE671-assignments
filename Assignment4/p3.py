import numpy as np
from scipy.optimize import minimize_scalar
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
    if points_im1.shape[0] < 4:
        raise ValueError('At least four point correspondences are required')

    measurement = np.vstack((points_im1[:, :2].T, points_im2[:, :2].T)).astype(float)
    measurement -= measurement.mean(axis=1, keepdims=True)
    U, singular_values, Vt = np.linalg.svd(measurement, full_matrices=False)
    print('Measurement matrix singular values:', singular_values)
    root_singular_values = np.diag(np.sqrt(singular_values[:3]))
    motion = U[:, :3] @ root_singular_values
    structure = root_singular_values @ Vt[:3, :]

    def quadratic_coefficients(first, second):
        return np.array([
            first[0] * second[0],
            first[0] * second[1] + first[1] * second[0],
            first[0] * second[2] + first[2] * second[0],
            first[1] * second[1],
            first[1] * second[2] + first[2] * second[1],
            first[2] * second[2]])

    constraints = []
    values = []
    for camera in range(2):
        first_row, second_row = motion[2*camera], motion[2*camera + 1]
        constraints.extend((quadratic_coefficients(first_row, first_row),
                            quadratic_coefficients(second_row, second_row),
                            quadratic_coefficients(first_row, second_row)))
        values.extend((1, 1, 0))

    coefficients = np.asarray(constraints)
    metric_parameters, _, _, _ = np.linalg.lstsq(coefficients, values, rcond=None)

    _, _, coefficient_right = np.linalg.svd(coefficients, full_matrices=True)
    coefficient_rank = np.linalg.matrix_rank(coefficients)
    null_space = coefficient_right[coefficient_rank:, :].T

    def metric_from_parameters(parameters):
        return np.array([[parameters[0], parameters[1], parameters[2]],
                         [parameters[1], parameters[3], parameters[4]],
                         [parameters[2], parameters[4], parameters[5]]])

    if null_space.shape[1] == 1:
        null_direction = null_space[:, 0]
        search_radius = 10 * max(np.linalg.norm(metric_parameters), 1e-3)
        candidates = np.linspace(-search_radius, search_radius, 1001)

        def metric_condition(offset):
            eigenvalues = np.linalg.eigvalsh(
                metric_from_parameters(metric_parameters + offset * null_direction))
            if eigenvalues[0] <= 0:
                return np.inf
            return eigenvalues[-1] / eigenvalues[0]

        candidate_conditions = np.array([metric_condition(offset) for offset in candidates])
        best_index = int(np.argmin(candidate_conditions))
        if np.isfinite(candidate_conditions[best_index]):
            lower = candidates[max(best_index - 1, 0)]
            upper = candidates[min(best_index + 1, len(candidates) - 1)]
            if lower < upper:
                fit = minimize_scalar(metric_condition, bounds=(lower, upper), method='bounded')
                metric_parameters = metric_parameters + fit.x * null_direction

    metric = metric_from_parameters(metric_parameters)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    if eigenvalues[0] > 0:
        upgrade = np.linalg.cholesky(metric)
    else:
        eigenvalue_floor = max(eigenvalues[-1] * 1e-2, np.finfo(float).eps)
        upgrade = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, eigenvalue_floor))) @ eigenvectors.T
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
