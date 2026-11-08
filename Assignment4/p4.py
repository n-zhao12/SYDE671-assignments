import sys
import numpy as np
import os
from scipy.optimize import least_squares
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.io import imread
from sfm_utils import *

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    rotations = (U @ W @ Vt, U @ W.T @ Vt)
    translation = U[:, 2]
    candidates = []
    for rotation in rotations:
        if np.linalg.det(rotation) < 0:
            rotation = -rotation
        candidates.extend((np.column_stack((rotation, translation)),
                           np.column_stack((rotation, -translation))))
    return np.asarray(candidates)

'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    image_points = np.asarray(image_points, dtype=float)
    camera_matrices = np.asarray(camera_matrices, dtype=float)
    if image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError('image_points must have shape Mx2')
    if camera_matrices.shape != (image_points.shape[0], 3, 4):
        raise ValueError('camera_matrices must have shape Mx3x4')

    equations = []
    for (x, y), camera in zip(image_points, camera_matrices):
        equations.extend((x * camera[2] - camera[0],
                          y * camera[2] - camera[1]))
    _, _, right = np.linalg.svd(np.asarray(equations))
    point = right[-1, :]
    if abs(point[3]) < 1e-12:
        raise ValueError('Triangulated point lies at infinity')
    return point[:3] / point[3]

'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2Mx1 reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    point_h = np.append(np.asarray(point_3d, dtype=float), 1)
    projected_h = camera_matrices @ point_h
    projected = projected_h[:, :2] / projected_h[:, 2, None]
    return (projected - image_points).reshape(-1)

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    point_h = np.append(np.asarray(point_3d, dtype=float), 1)
    rows = []
    for camera in camera_matrices:
        projected = camera @ point_h
        denominator = projected[2]
        if abs(denominator) < 1e-12:
            raise ValueError('Point projects to infinity in a camera')
        rows.append((camera[0, :3] * denominator -
                     projected[0] * camera[2, :3]) / denominator**2)
        rows.append((camera[1, :3] * denominator -
                     projected[1] * camera[2, :3]) / denominator**2)
    return np.asarray(rows)

'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    point = linear_estimate_3d_point(image_points, camera_matrices)
    for _ in range(10):
        error = reprojection_error(point, image_points, camera_matrices)
        point_jacobian = jacobian(point, camera_matrices)
        update, _, _, _ = np.linalg.lstsq(point_jacobian, error, rcond=None)
        point -= update
        if np.linalg.norm(update) < 1e-8:
            break
    return point

'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    image_points = np.asarray(image_points, dtype=float)
    if image_points.ndim == 2:
        image_points = image_points[None, ...]
    if image_points.ndim != 3 or image_points.shape[1:] != (2, 2):
        raise ValueError('image_points must have shape Nx2x2')

    candidates = estimate_initial_RT(E)
    first_camera = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    best_pose = None
    best_count = -1
    for pose in candidates:
        camera_matrices = np.asarray((first_camera, K @ pose))
        positive_depth = 0
        for pair in image_points:
            try:
                point = linear_estimate_3d_point(pair, camera_matrices)
            except ValueError:
                continue
            second_point = pose[:, :3] @ point + pose[:, 3]
            positive_depth += int(point[2] > 0 and second_point[2] > 0)
        if positive_depth > best_count:
            best_count = positive_depth
            best_pose = pose
    return best_pose

if __name__ == '__main__':
    run_pipeline = True

    # Load the data
    image_data_dir = 'data/statue/'
    unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    unit_test_image_matches = np.load('data/unit_test_image_matches.npy')
    image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        sorted(os.listdir('data/statue/images')) if '.jpg' in x]
    focal_length = 719.5459
    matches_subset = np.load(os.path.join(image_data_dir,
        'matches_subset.npy'), allow_pickle=True, encoding='latin1')[0,:]
    dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'), 
                               allow_pickle=True, encoding='latin1')
    fundamental_matrices = np.load(os.path.join(image_data_dir,
        'fundamental_matrices.npy'), allow_pickle=True, encoding='latin1')[0,:]

    # Part A: Computing the 4 initial R,T transformations from Essential Matrix
    print('-' * 80)
    print("Part A: Check your matrices against the example R,T")
    print('-' * 80)
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    E = K.T.dot(fundamental_matrices[0]).dot(K)
    im0 = imread(image_paths[0])
    im_height, im_width, _ = im0.shape
    example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
        [0.1019, 0.9948, 0.0045, -0.0089],
        [0.2041, -0.0254, 0.9786, 0.0331]])
    print("Example RT:\n", example_RT)
    estimated_RT = estimate_initial_RT(E)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part B: Determining the best linear estimate of a 3D point
    print('-' * 80)
    print('Part B: Check that the difference from expected point ')
    print('is near zero')
    print('-' * 80)
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        camera_matrices.copy())
    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    print("Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum())

    # Part C: Calculating the reprojection error and its Jacobian
    print('-' * 80)
    print('Part C: Check that the difference from expected error/Jacobian ')
    print('is near zero')
    print('-' * 80)
    estimated_error = reprojection_error(
            expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    print("Error Difference: ", np.fabs(estimated_error - expected_error).sum())
    expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
         [0., 154.33943931, 36.51165089],
         [141.87950588, -14.27738422, -56.20341644],
         [21.9792766, 149.50628901, 32.23425643]])
    print("Jacobian Difference: ", np.fabs(estimated_jacobian
        - expected_jacobian).sum())

    # Part D: Determining the best nonlinear estimate of a 3D point
    print('-' * 80)
    print('Part D: Check that the reprojection error from nonlinear method')
    print('is lower than linear method')
    print('-' * 80)
    estimated_3d_point_linear = linear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    error_linear = reprojection_error(
        estimated_3d_point_linear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Linear method error:", np.linalg.norm(error_linear))
    error_nonlinear = reprojection_error(
        estimated_3d_point_nonlinear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Nonlinear method error:", np.linalg.norm(error_nonlinear))

    # Part E: Determining the correct R, T from Essential Matrix
    print('-' * 80)
    print("Part E: Check your matrix against the example R,T")
    print('-' * 80)
    estimated_RT = estimate_RT_from_E(E,
        np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print("Example RT:\n", example_RT)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part F: Run the entire Structure from Motion pipeline
    if not run_pipeline:
        sys.exit()
    print('-' * 80)
    print('Part F: Run the entire SFM pipeline')
    print('-' * 80)
    frames = [0] * (len(image_paths) - 1)
    for i in range(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length,
                fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])
    merged_frame = merge_all_frames(frames)

    # Construct the dense matching
    camera_matrices = np.zeros((2,3,4))
    dense_structure = np.zeros((0,3))
    for i in range(len(dense_matches)):
        matches = dense_matches[i]
        camera_matrices[0,:,:] = merged_frame.K.dot(
            merged_frame.motion[i,:,:])
        camera_matrices[1,:,:] = merged_frame.K.dot(
                merged_frame.motion[i+1,:,:])
        points_3d = np.zeros((matches.shape[1], 3))
        use_point = np.array([True]*matches.shape[1])
        for j in range(matches.shape[1]):
            points_3d[j,:] = nonlinear_estimate_3d_point(
                matches[:,j].reshape((2,2)), camera_matrices)
        dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dense_structure[:,0], dense_structure[:,1], dense_structure[:,2],
        c='k', depthshade=True, s=2)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 10)
    ax.view_init(-100, 90)

    plt.show()
