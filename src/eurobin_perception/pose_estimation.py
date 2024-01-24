#!/usr/bin/env python3

"""
Contains the classes and functions that provide the functionalities needed
to estimate position and orientation.
"""

import time

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

## Temp:
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Point

## ----------------------------------------------------------------------
## Functions:
## ----------------------------------------------------------------------

def get_point_cloud_segments(pointcloud2_msg, detection_result_msg, camera_info_msg, 
                             cropped_pc_msg=None, cropped_pc_label='taskboard', 
                             debug=False):
    """
    Extracts the segments of a pointcloud that correspond to each object that was
    detected in an RGB image.
    Returns the list of points (coordinates) that fall within the field of view
    of each object's 2D bounding box.
    Optionally returns the "cropped" pointcloud of a given object (label).

    The function performs the following operations:
      - Converts a PointCloud2 message to a list of 3D point coordinates.
      - Back-projects all points onto the image plane using the camera's
        intrinsic parameters.
      - Checks whether each projected point falls within the boundaries of any
        of the given bounding boxes. If it does, the point is assigned to that
        bounding box's class/label.

    Parameters
    ----------
    pointcloud2_msg: sensor_msgs.PointCloud2
        PointCloud2 ROS message containing perceived point cloud
    detection_result_msg: eurobin_perception.BoundingBoxList
        Detection result ROS message containg list of bounding boxes
    camera_info_msg: sensor_msgs.CameraInfo
        Camera information ROS message
    cropped_pc_msg: sensor_msgs.PointCloud
        PointCloud ROS message to be populated with cropped point cloud
    cropped_pc_label: str
        Label of object for which to return the cropped point cloud
    debug: bool
        Whether to print some debugging messages

    Returns
    -------
    object_points_dict: dict
        Mapping between labels and lists of corresponding points (contained
        in objects of type sensor_msgs.point_cloud2.Point).
    cropped_pc_msg: sensor_msgs.PointCloud
        PointCloud ROS message, populated with cropped point cloud
    """

    # Get camera parameters:
    camera_P = camera_info_msg.P
    f_x = camera_P[0]; f_y = camera_P[5]
    c_x = camera_P[2]; c_y = camera_P[6]

    # Extract list of point positions (x, y, z) sensor_msgs/PointCloud2 message:
    read_points_start_time = time.time()
    pc_point_list = pc2.read_points_list(pointcloud2_msg, skip_nans=True, field_names=("x", "y", "z"))

    if debug:
        print('[DEBUG] [pose_estimator] Converted PC2 msg to points list in {:.2f}s'.format(time.time() - read_points_start_time))
    # TODO: Consider moving to ros_numpy (partial) vectorization if runtime must be improved.

    # Find and store the 3D points that fall within each detection BB using the 3D-to-2D back-projection method:
    object_points_dict = {}
    point_iter_start_time = time.time()
    for point in pc_point_list:
        # Run point containment check for each BB:
        for bbox in detection_result_msg.bounding_boxes:
            if bbox.label not in object_points_dict.keys():
                object_points_dict[bbox.label] = []

            # Compute projection of point onto image plane:
            # Note: point indices: {x, y, z} --> {0, 1, 2}.
            u = f_x * (point[0] / point[2]) + c_x
            v = f_y * (point[1] / point[2]) + c_y

            # Check if projected point is within the bounds of the BB:
            if u > bbox.xmin and u < bbox.xmax and v > bbox.ymin and v < bbox.ymax:
                object_points_dict[bbox.label].append(point)

                # Add the point to the "cropped point cloud" (visualization for debugging):
                if cropped_pc_msg is not None and bbox.label == cropped_pc_label:
                    cropped_pc_msg.points.append(Point(x=point[0], y=point[1], z=point[2]))

    if debug:
        print('[DEBUG] [pose_estimator] Iterated over all points for all labels in {:.2f}s'.format(time.time() - point_iter_start_time))

    return object_points_dict, cropped_pc_msg

def get_point_cloud_segments_v2(pc_points, bbox_dict_list, camera_params_dict, 
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
        Whether to print some debugging messages

    Returns
    -------
    object_points_dict: dict
        Mapping between labels and lists of corresponding points (contained
        in objects whose types will depend on the input pc_points, e.g.: 
        sensor_msgs.point_cloud2.Point, list, ndarray).
    cropped_pc_points: list
        Cropped point cloud set of 3D points
    """
    f_x, f_y = camera_params_dict['f_x'], camera_params_dict['f_y']
    c_x, c_y = camera_params_dict['c_x'], camera_params_dict['c_y']
    cropped_pc_points = []

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

                # Add the point to the "cropped point cloud" (visualization for debugging):
                if bbox['class'] == cropped_pc_label:
                    cropped_pc_points.append(list(point))

    if debug:
        print('[DEBUG] [pose_estimator] Iterated over all points for all labels in {:.2f}s'.format(time.time() - point_iter_start_time))

    return object_points_dict, cropped_pc_points

def estimate_object_positions(point_arrays_dict, debug=False):
    """
    Estimates the 3D positions of objects from lists of corresponding points.
    By default, positions are estimated by the mean of the points.

    Parameters
    ----------
    point_arrays_dict: dict
        Mapping between labels and ndarrays of corresponding points.
    debug: bool
        Whether to print some debugging messages

    Returns
    -------
    estimated_positions_dict: dict
        Mapping between labels and ndarrays containing 3D position coordinates
    """
    estimated_positions_dict = {}

    for label, points_array in point_arrays_dict.items():
        estimated_positions_dict[label] = points_array.mean(axis=0)
        if debug:
            print('[DEBUG] Label: {}'.format(label))
            print('[DEBUG] Points mean:', points_array.mean(axis=0))
            print('[DEBUG] Points std:', points_array.std(axis=0))
            print()

    return estimated_positions_dict 

def convert_object_points_to_arrays(point_lists_dict):
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
        point_arrays_dict[label] = np.stack(point_lists_dict[label])

    return point_arrays_dict 

def remove_outliers(points_array, percentiles=(25, 75), debug=False):
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
        Whether to print some debugging messages

    Returns
    -------
    points_array: ndarray
        3D position coordinates of a set of points post outlier removal
    """
    points_df = pd.DataFrame(data=points_array, columns=['x', 'y', 'z'])

    if debug:
        print('\n[DEBUG] Original points array shape:', points_array.shape)
        print(f'[DEBUG] percentiles: {percentiles}')
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
            print('[DEBUG] Removing outliers in {}:'.format(dim))
            print('[DEBUG] Number of remaining points:', len(points_df))

    points_array = np.array(points_df)

    return points_array

def remove_outliers_agglomerative(points_array, debug=False):
    """
    TODO

    Parameters
    ----------
    points_array: ndarray
        3D position coordinates of a set of points
    debug: bool
        Whether to print some debugging messages

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
        print('\n[DEBUG] Original points array shape:', points_array.shape)
        for cluster_id in range(n_clusters):
            print(f'[DEBUG] No. of points in cluster {cluster_id}: {num_points_dict[cluster_id]}')

        print(f'[DEBUG] Selecting cluster {inlier_cluster_index} as inlier')
        print(f'[DEBUG] Number of remaining points:', points_array.shape[0])

    return points_array


class PositionEstimator(object):
    """
    Object position estimator that utilizes CNN RGB detection results and
    depth data: point clouds.

    Note: this class was designed and is currently used for the taskboard
    challenge and dataset, but could be made more general in the future.

    Parameters
    ----------
        camera_params_dict: dict
            Camera parameter values: f_x, f_y, c_x, c_y
    """

    def __init__(self, camera_params_dict=None):
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
            Whether to print some debugging messages

        Returns
        -------
        object_points_dict: dict
            Mapping between labels and lists of corresponding points (contained
            in objects whose types will depend on the input pc_points, e.g.: 
            sensor_msgs.point_cloud2.Point, list, ndarray).
        cropped_pc_points: list
            Cropped point cloud set of 3D points
        """
        f_x, f_y = camera_params_dict['f_x'], camera_params_dict['f_y']
        c_x, c_y = camera_params_dict['c_x'], camera_params_dict['c_y']
        cropped_pc_points = []

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

                    # Add the point to the "cropped point cloud" (visualization for debugging):
                    if bbox['class'] == cropped_pc_label:
                        cropped_pc_points.append(list(point))

        if debug:
            iteration_time = time.time() - point_iter_start_time
            print(f'[DEBUG] [{self.name}] Iterated over all points for all labels in {iteration_time:.2f}s')

        return object_points_dict, cropped_pc_points

    def estimate_object_positions(self, bbox_dict_list, pc_point_list, 
                                  cropped_pc_label=None, debug=False):
        """
        TODO

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
            Whether to print some debugging messages

        Returns
        -------
        TODO
        """
        object_points_dict, cropped_pc_points = \
            self.get_point_cloud_segments(
                pc_points=pc_point_list,
                bbox_dict_list=bbox_dict_list,
                camera_params_dict=self.camera_params_dict,
                cropped_pc_label=cropped_pc_label,
                debug=debug
        )
        object_points_dict = convert_object_points_to_arrays(object_points_dict)

        if debug:
            # Copy unfiltered taskboard data for later visualizations:
            unfiltered_points_array = object_points_dict[cropped_pc_label].copy()
        else:
            unfiltered_points_array = None

        # Remove outliers using IQR method:
        for object_id, points_array in object_points_dict.items():
            if debug:
                print(f'\n[DEBUG] [{self.name}] Removing outliers from {object_id} points...')

            percentiles = (35, 65) if object_id == 'taskboard' \
                            else (25, 75)
            filtered_points_array = remove_outliers(points_array,
                                                    percentiles=percentiles,
                                                    debug=debug)
            object_points_dict[object_id] = filtered_points_array

        object_positions_dict = estimate_object_positions(object_points_dict, 
                                                          debug=debug)

        return (object_positions_dict, object_points_dict,
                cropped_pc_points, unfiltered_points_array) 

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
