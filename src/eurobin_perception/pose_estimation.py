#!/usr/bin/env python3

"""
Contains the classes and functions that provide the functionalities needed
to estimate position and orientation.
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# For debug visualizations:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from eurobin_perception.utils import _arrow3D, _annotate3D, minimum_bounding_rectangle, rotation_matrix_from_vectors
setattr(Axes3D, 'arrow3D', _arrow3D)
setattr(Axes3D, 'annotate3D', _annotate3D)


class PositionEstimator(object):
    """
    Object position estimator that utilizes CNN RGB detection results and
    depth data: point clouds.

    Note: at the moment, the class implements position estimation that should
    be general enough to work with any CNN detection results.
    A general orientation estimator is not included since it is usually
    problem-dependent. Instead, this class can be extended and custom orientation
    functions implemented

    Parameters
    ----------
    camera_params_dict: dict
        Camera parameter values: f_x, f_y, c_x, c_y
    """

    def __init__(self, camera_params_dict):
        self.camera_params_dict = camera_params_dict

        self.name = self.__class__.__name__

    def get_point_cloud_segments(self, pc_points, bbox_dict_list, 
                                 camera_params_dict, 
                                 cropped_pc_label='taskboard', debug=False):
        """
        Extracts the segments of a pointcloud's set of points that correspond 
        to each object that was detected in an RGB image.
        Returns the list of points (coordinates) that fall within the field of view
        of each object's 2D bounding box.
        Optionally returns the list of 3D points that form the  "cropped" 
        pointcloud of a given object (label).

        The function performs the following operations:
          - Back-projects all given 3D points onto the image plane using the camera's
            intrinsic parameters.
          - Checks whether each projected point falls within the boundaries of any
            of the given bounding boxes. If it does, the point is assigned to that
            bounding box's class/label.

        Parameters
        ----------
        pc_points: list or ndarray
            Iterable containing a set of 3D points. Can be a list of
            sensor_msgs.point_cloud2.Point or a numpy array of shape
            (num_points, 3).
        bbox_dict_list: 
            Dicts containing info on each detection bbox 
            (class, xmin, ymin, xmax, ymax, confidence)
        camera_params_dict: dict
            Camera parameter values: f_x, f_y, c_x, c_y
        cropped_pc_label: str
            Label of object for which to return the cropped point cloud (if debug)
        debug: bool
            Whether to print debugging messages

        Returns
        -------
        object_points_dict: dict
            Mapping between labels and lists of corresponding points (contained
            in objects whose types will depend on the input pc_points, e.g.: 
            sensor_msgs.point_cloud2.Point, list, ndarray).
        """
        f_x, f_y = camera_params_dict['f_x'], camera_params_dict['f_y']
        c_x, c_y = camera_params_dict['c_x'], camera_params_dict['c_y']

        # Find and store the 3D points that fall within each detection BB using the 3D-to-2D back-projection method:
        object_points_dict = {}
        point_iter_start_time = time.time()
        for point in pc_points:
            # Run point containment check for each BB:
            for bbox in bbox_dict_list:
                if bbox['class'] not in object_points_dict.keys():
                    object_points_dict[bbox['class']] = []

                # Compute projection of point onto image plane:
                # Note: point indices: {x, y, z} --> {0, 1, 2}.
                u = f_x * (point[0] / point[2]) + c_x
                v = f_y * (point[1] / point[2]) + c_y

                # Check if projected point is within the bounds of the BB:
                if u > bbox['xmin'] and u < bbox['xmax'] and v > bbox['ymin'] and v < bbox['ymax']:
                    object_points_dict[bbox['class']].append(point)

        if debug:
            iteration_time = time.time() - point_iter_start_time
            print(f'[DEBUG] [{self.name}] Iterated over all points for all labels in {iteration_time:.2f}s')

        return object_points_dict

    def estimate_object_positions(self, bbox_dict_list, pc_point_list, 
                                  cropped_pc_label=None, debug=False):
        """
        Returns estimates of the 3D positions of detected objects from:
            - CNN image object detections
            - a list of points from a point cloud

        Parameters
        ----------
        bbox_dict_list: 
            Dicts containing info on each detection bbox 
            (class, xmin, ymin, xmax, ymax, confidence)
        pc_points: list or ndarray
            Iterable containing a set of 3D points. Can be a list of
            sensor_msgs.point_cloud2.Point or a numpy array of shape
            (num_points, 3).
        cropped_pc_label: str
            Label of object for which to return the cropped point cloud (if debug)
        debug: bool
            Whether to print debugging messages and return cropped point cloud

        Returns
        -------
        object_positions_dict: dict
            Mapping between labels and estimated 3D positions (np.ndarrays)
        object_points_dict: dict
            Mapping between labels and lists of corresponding points (contained
            in objects whose types will depend on the input pc_points, e.g.: 
            sensor_msgs.point_cloud2.Point, list, np.ndarray).
        cropped_pc_points_array: ndarray
            Cropped point cloud set of 3D points (if debug)
        """
        object_points_dict = \
            self.get_point_cloud_segments(
                pc_points=pc_point_list,
                bbox_dict_list=bbox_dict_list,
                camera_params_dict=self.camera_params_dict,
                cropped_pc_label=cropped_pc_label,
                debug=debug
        )
        object_points_dict = self.convert_object_points_to_arrays(object_points_dict)

        if debug:
            # Copy unfiltered taskboard data for later visualizations:
            cropped_pc_points_array = object_points_dict[cropped_pc_label].copy()
        else:
            cropped_pc_points_array = None

        # Remove outliers using IQR method:
        for object_id, points_array in object_points_dict.items():
            if debug:
                print(f'\n[DEBUG] [{self.name}] Removing outliers from {object_id} points...')

            percentiles = (35, 65) if object_id == 'taskboard' else (25, 75)
            filtered_points_array = self.remove_outliers(points_array,
                                                         percentiles=percentiles,
                                                         debug=debug)
            object_points_dict[object_id] = filtered_points_array

        # Estimates the 3D positions of objects from lists of corresponding points.
        # By default, positions are estimated through the mean of points.
        object_positions_dict = {}
        for label, points_array in object_points_dict.items():
            object_positions_dict[label] = points_array.mean(axis=0)
            if debug:
                print(f'[DEBUG] [{self.name}] Label: {label}')
                print(f'[DEBUG] [{self.name}] Points mean: {points_array.mean(axis=0)}')
                print(f'[DEBUG] [{self.name}] Points std: {points_array.std(axis=0)}')
                print()

        # Retain copy for orientation estimation visualizations:
        self.cropped_pc_points_array = cropped_pc_points_array

        return (object_positions_dict, object_points_dict, cropped_pc_points_array) 

    def convert_object_points_to_arrays(self, point_lists_dict):
        """
        Converts the the values of the given dict from lists of sensor_msgs.Point
        to ndarrays containing all points of shape (num_points, 3).
        This simplifies a few downstream operations.

        Parameters
        ----------
        point_lists_dict: dict
            Mapping between labels and lists of corresponding points (contained
            in objects of type sensor_msgs.point_cloud2.Point).

        Returns
        -------
        point_arrays_dict: dict
            Mapping between labels and ndarrays of corresponding points.
        """
        point_arrays_dict = {}
        for label in point_lists_dict.keys():
            try:
                point_arrays_dict[label] = np.stack(point_lists_dict[label])
            except ValueError:
                print(f'[WARN] [{self.name}] No points to extract for label {label}!')
                print(f'[WARN] [{self.name}] Skipping label {label}...')
                continue

        return point_arrays_dict 

    def remove_outliers(self, points_array, percentiles=(25, 75), debug=False):
        """
        Removes outliers from the given array of 3D points using the IQR method.
        The percentiles that are used to compute the inter-quartile range can be
        configured.
        Assumes that the columns of points_array contain coodinates in the x, y,
        and z axes.

        Parameters
        ----------
        points_array: ndarray
            3D position coordinates of a set of points
        percentiles: tuple
            Upper and lower percentiles for the IQR computation
        debug: bool
            Whether to print debugging messages

        Returns
        -------
        points_array: ndarray
            3D position coordinates of a set of points post outlier removal
        """
        points_df = pd.DataFrame(data=points_array, columns=['x', 'y', 'z'])

        if debug:
            print(f'\n[DEBUG] [{self.name}] Original points array shape: {points_array.shape}')
            print(f'[DEBUG] [{self.name}] percentiles: {percentiles}')
        for dim in ['x', 'y', 'z']:
            lower_percentile = np.percentile(points_df[dim],
                                             percentiles[0],
                                             method='midpoint')
            upper_percentile = np.percentile(points_df[dim],
                                             percentiles[1],
                                             method='midpoint')
            IQR = upper_percentile - lower_percentile
            upper_bound = upper_percentile + (1.5 * IQR)
            lower_bound = lower_percentile - (1.5 * IQR)

            points_df = points_df.loc[(points_df[dim] >= lower_bound) & \
                                      (points_df[dim] <= upper_bound)]
            if debug:
                print(f'[DEBUG] [{self.name}] Removing outliers in {dim}:')
                print(f'[DEBUG] [{self.name}] Number of remaining points: {len(points_df)}')

        points_array = np.array(points_df)

        return points_array

    def load_camera_params(self, camera_params_dict):
        """
        Stores the given camera parameter values.

        Parameters
        ----------
        camera_params_dict: dict
            Camera parameter values: f_x, f_y, c_x, c_y

        Returns
        -------
        None
        """
        self.camera_params_dict = camera_params_dict

class TaskboardPoseEstimator(PositionEstimator):
    """
    Taskboard pose estimator that utilizes CNN RGB detection results and
    depth data: point clouds.

    It inherits position estimation functionalities from the PositionEstimator
    and implements a taskboard-specific orientation estimation algorithm.
    Note: this class was designed for the taskboard challenge.

    Parameters
    ----------
    camera_params_dict: dict
        Camera parameter values: f_x, f_y, c_x, c_y
    class_colors_dict: dict
        Map between class names and (R, G, B) color tuples (0-255)
    """
    def __init__(self, camera_params_dict, class_colors_dict):
        super().__init__(camera_params_dict)

        self.class_colors_dict = class_colors_dict

    def remove_outliers_agglomerative(self, points_array, debug=False):
        """
        Removes outliers from the given array of 3D points using the agglomerative
        clustering method.
        This has been found to significantly improve taskboard orientation estimation.

        Parameters
        ----------
        points_array: ndarray
            3D position coordinates of a set of points
        debug: bool
            Whether to print debugging messages

        Returns
        -------
        points_array: ndarray
            3D position coordinates of a set of points post outlier removal
        """
        n_clusters = 2
        model = AgglomerativeClustering(n_clusters=n_clusters,
                                        linkage='single',
                                        metric='cityblock')
        point_indices = model.fit_predict(points_array)
        cluster_points = dict(zip(np.arange(n_clusters),
                                  [points_array[point_indices == cluster_id] \
                                   for cluster_id in range(n_clusters)]))
        num_points_dict = dict(zip(list(cluster_points.keys()),
                                   [cluster_points[cluster_id].shape[0] \
                                    for cluster_id in range(n_clusters)]))

        # Select cluster with larger number of points as inlier:
        inlier_cluster_index = np.argmax(list(num_points_dict.values()))

        points_array = cluster_points[inlier_cluster_index]

        if debug:
            print(f'\n[DEBUG] [{self.name}] Original points array shape: {points_array.shape}')
            for cluster_id in range(n_clusters):
                print(f'[DEBUG] [{self.name}] No. of points in cluster {cluster_id}: {num_points_dict[cluster_id]}')

            print(f'[DEBUG] [{self.name}] Selecting cluster {inlier_cluster_index} as inlier')
            print(f'[DEBUG] [{self.name}] Number of remaining points: {points_array.shape[0]}')

        return points_array

    def estimate_tb_orientation(self, tb_points_array, object_positions_dict, debug=False, **kwargs):
        """
        Implements and algorithm that estimates the orientation of the taskboard, given
        point cloud that have been classified according to the taskboard classes.
        This is achieved by:
            - Filtering the taskboard points using agglomerative clustering
            - Computing the normal to the best-fit plane on these points
              (This normal represents the taskboard's z-axis.)
            - Projecting the points on to the best-fit plane
            - Fitting a minimum-bounding rectangle around the projected points
            - Matching the corners of the fit rectangle to the known corners of the taskboard,
              using the provided, estimated locations of its parts.
              (e.g., buttons are near top-right, LCD is near bottom-right, etc.)
            - Computing the orientations of the vertical and horizontal sides of the taskboard
              using the positions of the identified corners.
              (These two vectors represents the taskboard's x and y axes.)
            - Combining the three comouted vectors to form a matrix that describes the transformation
              between the camera and taskboard frames, i.e. the taskboard's relative orientation.

        Optionally, the results of each stage are visualized on a 3D plot.

        Parameters
        ----------
        tb_points_array: ndarray
            3D position coordinates of set of points belonging to class "taskboard"
        object_positions_dict: dict
            Mapping between labels and estimated 3D positions (np.ndarrays)
        debug: bool
            Whether to print debugging messages and visualize results in 3D plots

        Returns
        -------
        tb_tf_matrix: ndarray
            Matrix describing the homogeneous TF between the camera and taskboard frames
        orientation_estimation_success: bool
            Whether orientation was successfully estimated
        vertical_side_found: bool
            Whether the vertical side(s) of the taskboard could be successfully recognized
        horizontal_side_found: bool
            Whether the vertical side(s) of the taskboard could be successfully recognized
        """
        vertical_side_found, horizontal_side_found = False, False

        ## ----------------------------------------
        ## Data Loading and Preprocessing:
        ## ----------------------------------------

        # Get 3D point cloud segment corresponding to the taskboard:
        tb_position = object_positions_dict['taskboard']

        print(f'[INFO] [{self.name}] Removing TB PC outliers using agglomerative clustering...', flush=True)
        tb_pc_outlier_removal_start_time = time.time()
        tb_points_array = self.remove_outliers_agglomerative(tb_points_array, debug=debug)
        if debug:
            elapsed_time = time.time() - tb_pc_outlier_removal_start_time
            print(f'[DEBUG] [{self.name}] Removed TB PC outliers in {elapsed_time:.2f}s')

        ## ----------------------------------------
        ## Extracting Normal of Best-Fit Plane:
        ## ----------------------------------------

        # Get best-fit plane normal:
        eigenvectors_eigh = np.linalg.eigh(np.cov(tb_points_array.T))[1]
        plane_normal_eigenvector = eigenvectors_eigh[:, 0]
        # If z coordinate is negative, invert the vector to rectify the resultant axes:
        if plane_normal_eigenvector[2] < 0.:
            plane_normal_eigenvector = -plane_normal_eigenvector 

        ## ----------------------------------------
        ## Fitting Minimum Bounding Rectangle on Planar Data:
        ## ----------------------------------------

        # Estimate minimum bounding rectangle on data projected onto best-fit plane:
        rot_matrix = rotation_matrix_from_vectors([0, 0, 1], plane_normal_eigenvector)
        points_oriented_array = tb_points_array @ rot_matrix

        rect_oriented_corners = minimum_bounding_rectangle(points_oriented_array[:, :2])
        mid_point_oriented = points_oriented_array.mean(axis=0)

        rect_oriented_corners_aug = np.hstack((rect_oriented_corners,
                                      np.ones((rect_oriented_corners.shape[0], 1)) * mid_point_oriented[2]))
        rect_rectified_corners = rect_oriented_corners_aug @ rot_matrix.T

        ## ----------------------------------------
        ## Estimating Taskboard Orientation:
        ## ----------------------------------------

        # Localize quadrant corners using known detected objects:
        object_nearest_corner_dict = {}
        for object_id, position_array in object_positions_dict.items():
            corner_id = np.linalg.norm(rect_rectified_corners - position_array, axis=1).argmin()
            object_nearest_corner_dict[object_id] = corner_id

        # Quadrant identification algorithm:
        quadrant_corner_ids = {}
        if 'red_button' in object_nearest_corner_dict.keys():
            quadrant_corner_ids['1'] = object_nearest_corner_dict['red_button']
        elif 'blue_button' in object_nearest_corner_dict.keys():
            quadrant_corner_ids['1'] = object_nearest_corner_dict['blue_button']
        elif 'multimeter_connector' in object_nearest_corner_dict.keys():
            quadrant_corner_ids['1'] = object_nearest_corner_dict['multimeter_connector']
        else:
            print(f'[INFO] [{self.name}] Could not locate quadrant 1!', flush=True)
            quadrant_corner_ids['1'] = None

        if 'hatch_handle' in object_nearest_corner_dict.keys():
            quadrant_corner_ids['2'] = object_nearest_corner_dict['hatch_handle']
        elif 'multimeter_probe' in object_nearest_corner_dict.keys():
            quadrant_corner_ids['2'] = object_nearest_corner_dict['multimeter_probe']
        else:
            print(f'[INFO] [{self.name}] Could not locate quadrant 2!', flush=True)
            quadrant_corner_ids['2'] = None

        if 'lcd' in object_nearest_corner_dict.keys():
            quadrant_corner_ids['4'] = object_nearest_corner_dict['lcd']
        elif 'slider' in object_nearest_corner_dict.keys():
            quadrant_corner_ids['4'] = object_nearest_corner_dict['slider']
        else:
            print(f'[INFO] [{self.name}] Could not locate quadrant 3!', flush=True)
            quadrant_corner_ids['4'] = None

        if quadrant_corner_ids['4'] == None or \
                quadrant_corner_ids['4'] == quadrant_corner_ids['1'] or \
                quadrant_corner_ids['4'] == quadrant_corner_ids['2']:
            print(f'[WARN] [{self.name}] Using test heuristic to determine quadrant 4...', flush=True)
            q1_coordinates = rect_rectified_corners[quadrant_corner_ids['1']]
            q2_coordinates = rect_rectified_corners[quadrant_corner_ids['2']]
            corner_distances = np.linalg.norm(rect_rectified_corners - q1_coordinates, axis=1)
            corner_distances[quadrant_corner_ids['1']] = np.inf
            corner_distances[quadrant_corner_ids['2']] = np.inf

            quadrant_corner_ids['4'] = corner_distances.argmin()

        quadrant_corner_ids['3'] = None
        corner_quadrant_ids = dict(zip(quadrant_corner_ids.values(), quadrant_corner_ids.keys()))

        rect_corners = rect_rectified_corners.copy()

        if debug:
            print(f'\n[DEBUG] [{self.name}] Rectangle Corners:\n{rect_corners}')
            # print(f'[DEBUG] [{self.name}] Quadrant Corner ID Candidates:\n{quadrant_corner_id_candidates}')

            print(f'\n[DEBUG] [{self.name}] Quadrant corner IDs:\n{quadrant_corner_ids}')
            print(f'[DEBUG] [{self.name}] Corner quadrant IDs:\n{corner_quadrant_ids}')

            print(f'\n[DEBUG] [{self.name}] Estimated best-fit plane normal vector:\n{plane_normal_eigenvector}')

        # Estimate first orientation vector:
        if quadrant_corner_ids['1'] is not None and quadrant_corner_ids['2'] is not None:
            orientation_vector_2 = rect_corners[quadrant_corner_ids['1'], :] - rect_corners[quadrant_corner_ids['2'], :]
            horizontal_side_found = True
        elif quadrant_corner_ids['3'] is not None and quadrant_corner_ids['4'] is not None:
            orientation_vector_2 = rect_corners[quadrant_corner_ids['3'], :] - rect_corners[quadrant_corner_ids['4'], :]
            horizontal_side_found = True
        else:
            print(f'[WARN] [{self.name}] Can not determine top/bottom of board!', flush=True)
            orientation_vector_2 = rect_corners[2, :] - rect_corners[1, :]

        # Estimate first orientation vector:
        if quadrant_corner_ids['1'] is not None and quadrant_corner_ids['4'] is not None:
            orientation_vector_1 = rect_corners[quadrant_corner_ids['1'], :] - rect_corners[quadrant_corner_ids['4'], :]
            vertical_side_found = True
        elif quadrant_corner_ids['2'] is not None and quadrant_corner_ids['3'] is not None:
            orientation_vector_1 = rect_corners[quadrant_corner_ids['2'], :] - rect_corners[quadrant_corner_ids['3'], :]
            vertical_side_found = True
        else:
            print(f'[WARN] [{self.name}] Can not determine right/left side of board!', flush=True)
            orientation_vector_1 = rect_corners[1, :] - rect_corners[0, :]

        if debug:
            print(f'\n[DEBUG] [{self.name}] Orientation vector 1 (unnormalized):\n{orientation_vector_1}')
            print(f'[DEBUG] [{self.name}] Orientation vector 2 (unnormalized):\n{orientation_vector_2}')

        orientation_vectors = np.stack((orientation_vector_1 / np.linalg.norm(orientation_vector_1),
                                        orientation_vector_2 / np.linalg.norm(orientation_vector_2)))

        tb_orientation_matrix = np.vstack((orientation_vectors, plane_normal_eigenvector))

        # Verify that the estimated orientation vectors are both perpendicular to the plane normal vector:
        if not all((orientation_vectors[:, :] @ plane_normal_eigenvector) < 1e-10):
            print(f'[INFO] [{self.name}] Estimated orientation vectors are not orthogonal! Orientation matrix:\n {tb_orientation_matrix}', flush=True)
            print(f'[INFO] [{self.name}] Will re-attempt to estimate orientation...', flush=True)

            return None, False, vertical_side_found, horizontal_side_found

        # Re-orient axes for desired convention:
        reorientation_matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        tb_orientation_matrix = reorientation_matrix @ tb_orientation_matrix

        tb_tf_matrix = np.hstack((np.vstack((tb_orientation_matrix, np.zeros(3))), np.array([[0, 0, 0, 1]]).T))
        # tb_tf_matrix = np.hstack((np.vstack((tb_orientation_matrix, np.zeros(3))), np.array([[tb_position[0], tb_position[1], tb_position[2], 1]]).T))

        if debug:
            print(f'\n[DEBUG] [{self.name}] Orientation vector 1 (normalized):\n{orientation_vectors[0, :]}')
            print(f'[DEBUG] [{self.name}] Orientation vector 2 (normalized):\n{orientation_vectors[1, :]}')

            print(f'\n[DEBUG] [{self.name}] Taskboard orientation matrix:\n{tb_orientation_matrix}')
            print(f'[DEBUG] [{self.name}] Taskboard surface tb_position position:\n{tb_position}')

        orientation_estimation_success = True if vertical_side_found and horizontal_side_found else False

        ## ----------------------------------------
        ## Visualizing Results:
        ## ----------------------------------------

        if debug:
            fig = plt.figure('Original Point Cloud Data')
            ax = fig.add_subplot(projection='3d')
            ax.scatter(self.cropped_pc_points_array[:, 0], self.cropped_pc_points_array[:, 1],
                       self.cropped_pc_points_array[:, 2],
                       label='Original', c='tab:blue', s=0.01, alpha=1.0)
            ax.legend()
            ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_zlabel('$z$')
            xlims, ylims, zlims  = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
            ax.view_init(-92, -86)

            fig = plt.figure('Taskboard Orientation Estimation Results')
            ax = fig.add_subplot(projection='3d')

            if not kwargs['hide_pc_points']:
                ax.scatter(tb_points_array[:, 0], tb_points_array[:, 1], tb_points_array[:, 2],
                           label='Points', c='black', s=0.01, alpha=0.2)

            # Best-fit plane's normal vector:
            end_point = plane_normal_eigenvector * kwargs['arrow_scale_factor']
            ax.arrow3D(*tb_position, *end_point, mutation_scale=10)
            ax.annotate3D('Plane Normal',
                          xyz=end_point + tb_position + (kwargs['arrow_scale_factor'] * .0),
                          xytext=(0., 0.),
                          textcoords='offset points',
                          ha='left', va='bottom')

            # Visualize best-fit plane representation:
            plane_bounds = list(zip(tb_points_array.min(axis=(0))[:2],
                                    tb_points_array.max(axis=(0))[:2]))
            xx, yy = np.meshgrid(np.linspace(plane_bounds[0][0], plane_bounds[0][1], 10),
                                 np.linspace(plane_bounds[1][0], plane_bounds[1][1], 10))
            A, B, C = plane_normal_eigenvector
            D = -tb_position @ plane_normal_eigenvector
            zz = (-(A * xx) - (B * yy) - D) / C
            ax.plot_surface(xx, yy, zz, alpha=0.3, facecolor='grey',
                            color='black', cstride=3, rstride=3,)
            ax.annotate3D('Best-fit Plane',
                          xyz=(plane_bounds[0][1], plane_bounds[1][1], zz.max()),
                          xytext=(0., 0.), textcoords='offset points',
                          ha='left', va='bottom')

            # Visualized fitted and rectified 2D rectangles:
            if kwargs['plot_fitted_rectangle']:
                rect_corners_naive = minimum_bounding_rectangle(tb_points_array[:, :2])
                rect_corners_aug = np.vstack((rect_corners_naive, rect_corners_naive[0, :]))
                rect_corners_aug = np.hstack((rect_corners_aug,
                                          np.ones((rect_corners_aug.shape[0], 1)) * tb_position[2]))
                ax.plot(rect_corners_aug[:, 0], rect_corners_aug[:, 1], rect_corners_aug[:, 2],
                         color='black', label='Best-Fit Rectangle (Naive)')

            if kwargs['plot_fitted_rectified_rectangle']:
                rect_corners_aug = np.vstack((rect_rectified_corners, rect_rectified_corners[0, :]))
                ax.plot(rect_corners_aug[:, 0], rect_corners_aug[:, 1], rect_corners_aug[:, 2],
                         color='pink', label='Best-Fit Rectangle (Rectified)')

            # Annotate quadrant points:
            for corner_id, quadrant_id in corner_quadrant_ids.items():
                if corner_id is None:
                    continue

                corner_point = rect_rectified_corners[corner_id, :]
                ax.scatter(*corner_point, c='black', marker='s')
                ha = 'right' if corner_point[0] <= 0 else 'left'
                va = 'top' if corner_point[1] <= 0 else 'bottom'
                ax.annotate3D('Q. {}'.format(quadrant_id),
                              xyz=corner_point + (corner_point * 0.05),
                              xytext=(0., 0.), textcoords='offset points',
                              ha=ha, va=va)

            # Annotate top or bottom side of TB (if possible):
            if quadrant_corner_ids['1'] is not None and quadrant_corner_ids['2'] is not None:
                dist = rect_corners[quadrant_corner_ids['1'], :] - rect_corners[quadrant_corner_ids['2'], :]
                annotation_point = np.array([rect_corners[quadrant_corner_ids['2'], 0] + (dist[0] / 2.),
                                             rect_corners[quadrant_corner_ids['2'], 1] + (dist[1] / 2.),
                                             rect_corners[quadrant_corner_ids['2'], 2] + (dist[2] / 2.)])
                ax.annotate3D('Top',
                              xyz=annotation_point + (annotation_point * 0.15),
                              xytext=(0., 0.), textcoords='offset points',
                              ha='center', va='top')
            elif quadrant_corner_ids['3'] is not None and quadrant_corner_ids['4'] is not None:
                dist = rect_corners[quadrant_corner_ids['4'], :] - rect_corners[quadrant_corner_ids['3'], :]
                annotation_point = np.array([rect_corners[quadrant_corner_ids['3'], 0] + (dist[0] / 2.),
                                             rect_corners[quadrant_corner_ids['3'], 1] + (dist[1] / 2.),
                                             rect_corners[quadrant_corner_ids['3'], 2] + (dist[2] / 2.)])
                ax.annotate3D('Bottom',
                              xyz=annotation_point + (annotation_point * 0.15),
                              xytext=(0., 0.), textcoords='offset points',
                              ha='center', va='top')

            # Annotate left or right side of TB (if possible):
            if quadrant_corner_ids['1'] is not None and quadrant_corner_ids['4'] is not None:
                dist = rect_corners[quadrant_corner_ids['1'], :] - rect_corners[quadrant_corner_ids['4'], :]
                annotation_point = np.array([rect_corners[quadrant_corner_ids['4'], 0] + (dist[0] / 2.),
                                             rect_corners[quadrant_corner_ids['4'], 1] + (dist[1] / 2.),
                                             rect_corners[quadrant_corner_ids['4'], 2] + (dist[2] / 2.)])
                ax.annotate3D('Right',
                              xyz=annotation_point + (annotation_point * 0.15),
                              xytext=(0., 0.), textcoords='offset points',
                              ha='center', va='top')
            elif quadrant_corner_ids['2'] is not None and quadrant_corner_ids['3'] is not None:
                dist = rect_corners[quadrant_corner_ids['2'], :] - rect_corners[quadrant_corner_ids['3'], :]
                annotation_point = np.array([rect_corners[quadrant_corner_ids['3'], 0] + (dist[0] / 2.),
                                             rect_corners[quadrant_corner_ids['3'], 1] + (dist[1] / 2.),
                                             rect_corners[quadrant_corner_ids['3'], 2] + (dist[2] / 2.)])
                ax.annotate3D('Left',
                              xyz=annotation_point + (annotation_point * 0.15),
                              xytext=(0., 0.), textcoords='offset points',
                              ha='center', va='top')

            # Visualize orientation vectors:
            for dim in range(2):
                end_point = orientation_vectors[dim, :] * kwargs['arrow_scale_factor']
                side_found = horizontal_side_found if dim else vertical_side_found
                head_width = 0.01 if side_found else 0.
                arrowstyle = '->' if side_found else '-'

                ax.arrow3D(*tb_position, *end_point, mutation_scale=10, arrowstyle=arrowstyle)

            # Visualized detected objects' positions:
            for object_id, position_array in object_positions_dict.items():
                ax.scatter(*position_array,
                           c=[np.array(self.class_colors_dict[object_id]) / 255.],
                           label='Object: {}'.format(object_id))

            ax.legend(prop={'size': 7}, bbox_to_anchor=(1.03, 1.0))
            ax.set_xlabel('$x$'), ax.set_ylabel('$y$'), ax.set_zlabel('$z$')
            ax.set_xlim(np.array(xlims) * 1.0)
            ax.set_ylim(np.array(ylims) * 1.0)
            ax.set_zlim(np.array(zlims) * 1.0)
            ax.view_init(-92, -86)

            plt.show()
    
        return tb_tf_matrix, orientation_estimation_success, vertical_side_found, horizontal_side_found

