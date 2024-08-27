#!/usr/bin/env python3

"""
Estimates the poses of objects from image bounding boxes (BBs) and depth data.

Note: this component was designed for and tested with an Intel Realsense D435
sensor. In particular, it expects messages to be published on the following 
topics (configurable through launch parameters):
    - /camera/color/image_raw (Image)
    - /camera/depth/color/points (PointCloud2)
    - /camera/color/camera_info  (CameraInfo)
"""

import os
import sys
import time
import copy
import pickle
import socket

import rclpy
import tf2_ros
import numpy as np
# import sensor_msgs.point_cloud2 as pc2
# Note: current ROS2 port of point_cloud2:
import sensor_msgs_py.point_cloud2 as pc2

from rclpy import Parameter
from rclpy.node import Node
# from tf.transformations import quaternion_from_matrix
from tf_transformations import quaternion_from_matrix
from launch_ros.substitutions import FindPackageShare

from geometry_msgs.msg import Point, Point32, Pose, Quaternion, Vector3, TransformStamped
from sensor_msgs.msg import PointCloud, PointCloud2, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from tum_tb_perception_msgs.msg import BoundingBoxList, ObjectList, Object

from tum_tb_perception.pose_estimation import TaskboardPoseEstimator
from tum_tb_perception.utils import bbox_list_msg_to_list, obj_list_msg_to_json
from tum_tb_perception.visualization import load_class_color_map
from tum_tb_perception.dataset import load_labels


## ----------------------------------------------------------------------
## ROS Nodes, Callbacks and Message Initializations:
## ----------------------------------------------------------------------

class PoseEstimatorNode(Node):

    def __init__(self):
        super().__init__('pose_estimator')

        self.pkg_share_path = FindPackageShare(package='tum_tb_perception').find('tum_tb_perception')

        # Get node parameters:
        self.declare_parameter('class_colors_file_path', 
            os.path.join(self.pkg_share_path, 'config/class_colors_taskboard.yaml'))
        self.declare_parameter('output_dir_path', '/tmp')
        self.declare_parameter('taskboard_frame_name', 'taskboard_frame')
        self.declare_parameter('num_retries', 3)
        self.declare_parameter('udp_ip', 'localhost')
        self.declare_parameter('udp_output_port', 6000)
        self.declare_parameter('pointcloud_topic', '/camera/depth/color/points')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('detector_result_topic', '/tum_tb_perception/detection_result')
        self.declare_parameter('object_positions_pub_topic', '/tum_tb_perception/object_positions')
        self.declare_parameter('object_poses_pub_topic', '/tum_tb_perception/object_poses')
        self.declare_parameter('object_marker_pub_topic', '/tum_tb_perception/object_markers')
        self.declare_parameter('cropped_pc_pub_topic', '/tum_tb_perception/cropped_pc')
        self.declare_parameter('labels_file_path', 'config/labels.txt')
        self.declare_parameter('save_output', False)
        self.declare_parameter('rate', 10)
        self.declare_parameter('debug', False)

        self.class_colors_file_path = self.get_parameter('class_colors_file_path').value
        self.output_dir_path = self.get_parameter('output_dir_path').value
        self.taskboard_frame_name = self.get_parameter('taskboard_frame_name').value
        self.num_retries = self.get_parameter('num_retries').value
        self.udp_ip = self.get_parameter('udp_ip').value
        self.udp_output_port = self.get_parameter('udp_output_port').value
        self.pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.detector_result_topic = self.get_parameter('detector_result_topic').value
        self.object_positions_pub_topic = self.get_parameter('object_positions_pub_topic').value
        self.object_poses_pub_topic = self.get_parameter('object_poses_pub_topic').value
        self.object_marker_pub_topic = self.get_parameter('object_marker_pub_topic').value
        self.cropped_pc_pub_topic = self.get_parameter('cropped_pc_pub_topic').value
        self.labels_file_path = self.get_parameter('labels_file_path').value
        self.save_output = self.get_parameter('save_output').value
        self.rate = self.get_parameter('rate').value
        self.debug = self.get_parameter('debug').value

        # Initialize subscribers:
        self.pc2_subscription = self.create_subscription(PointCloud2,
                                                         self.pointcloud_topic,
                                                         self.pc2_callback,
                                                         10)
        self.detector_subscription = self.create_subscription(BoundingBoxList,
                                                              self.detector_result_topic,
                                                              self.detection_callback,
                                                              10)
        self.camera_info_subscription = self.create_subscription(CameraInfo,
                                                                 self.camera_info_topic,
                                                                 self.camera_info_callback,
                                                                 10)

        # Initialize publishers:
        self.object_positions_publisher = self.create_publisher(ObjectList, self.object_positions_pub_topic, 10)
        self.object_poses_publisher = self.create_publisher(ObjectList, self.object_poses_pub_topic, 10)
        self.object_marker_publisher = self.create_publisher(MarkerArray, self.object_marker_pub_topic, 10)
        if self.debug:
            self.cropped_pc_publisher = self.create_publisher(PointCloud, self.cropped_pc_pub_topic, 10)

        # Initialize messages:
        self.current_pc2_msg = None
        self.current_detection_msg = None
        self.current_camera_info_msg = None

        # Currently unused:
        # # Initialize timer:
        # self.timer = self.create_timer(float(1. / self.rate), self.timer_callback)
        
        # Initialize data variables:
        self.initialize()

        # self.get_logger().info(f'Waiting for trigger message...\n')

    def initialize(self):
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        ## ----------------------------------------------------------------------
        ## UDP Initializations
        ## ----------------------------------------------------------------------

        self.get_logger().info(f'Initializing UDP socket with address ' + \
                               f'family AF_INET and type SOCK_DGRAM')
        self.get_logger().info(f'Will send messages over IP {self.udp_ip} ' + \
                               f'and port {self.udp_output_port}.')
        self.udp_output_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        ## ----------------------------------------------------------------------
        ## Estimator Initialization
        ## ----------------------------------------------------------------------

        self.get_logger().info(f'Initializing TaskboardPoseEstimator...')
        # Initialize TaskboardPoseEstimator (with current camera parameters):
        self.class_colors_dict = load_class_color_map(self.class_colors_file_path)
        self.position_estimator = TaskboardPoseEstimator(class_colors_dict=self.class_colors_dict)

        # Set up output data directory:
        if self.save_output:
            output_sub_dir_path = 'pose_estimator_output_' + \
                                  datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir_path = os.path.join(self.output_dir_path, output_sub_dir_path)
            self.get_logger().info(f'Saving output data in ' + \
                                   f'{output_dir_path}')

            if not os.path.isdir(output_dir_path):
                self.get_logger().info(f'Output directory does not exist! ' + \
                                       f'Creating now...')
                os.makedirs(output_dir_path)

        # Currently unused:
        # self.current_time = time.time()
        # self.timer_counter = 0.
        # self.rate_object = self.create_rate(self.rate)

        self.object_marker_id = 0

        self.labels_list = load_labels(self.labels_file_path)

    ## ----------------------------------------------------------------------
    ## Callback Definitions
    ## ----------------------------------------------------------------------

    def pc2_callback(self, msg):
        self.current_pc2_msg = msg

    def detection_callback(self, msg):
        self.current_detection_msg = msg

    def camera_info_callback(self, msg):
        self.current_camera_info_msg = msg

        # if pose_estimator.debug:
        #     self.get_logger().info(f'[DEBUG] Received image message')
        #     self.get_logger().info(f'[DEBUG] {self.current_image_msg.height}')
        #     self.get_logger().info(f'[DEBUG] {self.current_image_msg.width}')
        #     self.get_logger().info(f'[DEBUG] {self.current_image_msg.header.frame_id}')

    ## ----------------------------------------------------------------------
    ## Helper Function Definitions
    ## ----------------------------------------------------------------------

    def timer_callback(self):
        pass

    def save_data(self):
        # TODO: Move data saving here
        raise NotImplementedError

    def get_camera_params_dict(self):
        """
        Retrieves and returns camera parameters in a dict.

        Parameters
        -------
        None

        Returns
        -------
        camera_params_dict: dict
            Camera intrinsic parameters (P matrix elements)
        """
        f_x = self.current_camera_info_msg.p[0]; f_y = self.current_camera_info_msg.p[5]
        c_x = self.current_camera_info_msg.p[2]; c_y = self.current_camera_info_msg.p[6]

        return {'f_x': f_x, 'f_y': f_y, 'c_x': c_x, 'c_y': c_y}

    ## ----------------------------------------------------------------------
    ## RViz Function Definitions
    ## ----------------------------------------------------------------------

    def get_point_markers(self, point, frame_id, label='', color_value=(0., 0., 0.)):
        """
        Creates RViz position and text markers for a given 3D point.

        Parameters
        ----------
        point: ndarray
            3D position coordinates
        frame_id: str
            Name of given point's coordinate frame
        label: str
            Text with which the point will be labeled
        color_value: tuple
           (R, G, B) values for the point and text markers [0., 255.] 

        Returns
        -------
        marker_msg: visualization_msgs.Marker 
            Point visualization marker ROS message
        text_marker_msg: visualization_msgs.Marker 
            Text visualization marker ROS message
        """
        marker_msg = Marker()
        marker_msg.header.frame_id = frame_id
        marker_msg.id = self.object_marker_id
        marker_msg.type = 2                                       # Sphere
        marker_msg.action = 0                                     # Add/modify
        # Note: ROS2 geometry_msgs/Point accepts only floats, not numpy.float64, hence the cast:
        marker_msg.pose = Pose(position=Point(**dict(zip(['x', 'y', 'z'], 
                                                         [float(value) for value in point]))),
                               orientation=Quaternion(**dict(zip(['x', 'y', 'z', 'w'], (0., 0., 0., 1.)))))
        marker_msg.scale.x = 0.01
        marker_msg.scale.y = 0.01
        marker_msg.scale.z = 0.01
        marker_msg.color.r = color_value[0] / 255.
        marker_msg.color.g = color_value[1] / 255.
        marker_msg.color.b = color_value[2] / 255.
        marker_msg.color.a = 1.0
        self.object_marker_id += 1

        text_marker_msg = copy.deepcopy(marker_msg)
        text_marker_msg.id = self.object_marker_id
        text_marker_msg.type = 9                                  # Text-view-facing
        text_marker_msg.text = label
        text_marker_msg.pose.position.x = marker_msg.pose.position.x + \
                                          (0.005 * (len(label) / 2.))
        text_marker_msg.pose.position.y = marker_msg.pose.position.y - 0.01
        self.object_marker_id += 1

        return marker_msg, text_marker_msg

    def clear_object_markers(self):
        """
        Clears all current RViz object markers.

        Parameters
        -------
        None

        Returns
        -------
        None
        """
        marker_array_msg = MarkerArray()

        marker_msg = Marker()
        marker_msg.id = self.object_marker_id 
        marker_msg.action = 3                                   # Delete all

        marker_array_msg.markers.append(marker_msg)
        self.object_marker_publisher.publish(marker_array_msg)

        # Reset object marker ID:
        self.object_marker_id = 0

    def run_node(self):
        try:
            while rclpy.ok():
                rclpy.spin_once(self)

                if self.current_detection_msg is not None:
                    self.clear_object_markers()
                    if self.current_pc2_msg is not None:

                        ## ----------------------------------------------------------------------
                        ## Position Estimation:
                        ## ----------------------------------------------------------------------

                        self.get_logger().info(f'Received detection result message.')
                        self.get_logger().info(f'Estimating detected object positions...')
                        position_estimation_start_time = time.time()

                        # Extract list of point positions (x, y, z) from sensor_msgs/PointCloud2 message:
                        read_points_start_time = time.time()
                        pc_point_list = pc2.read_points_list(self.current_pc2_msg, 
                                                             skip_nans=True, 
                                                             field_names=("x", "y", "z"))
                        if self.debug:
                            self.get_logger().info(f'[DEBUG] Converted PC2 msg to points list in {time.time() - read_points_start_time:.2f}s')
                        # TODO: Consider moving to ros_numpy (partial) vectorization if runtime must be improved.

                        # Get CNN detections and convert to list of dicts:
                        bbox_dict_list = bbox_list_msg_to_list(self.current_detection_msg)

                        # Estimate object positions:
                        object_positions_dict, object_points_dict, cropped_pc_points_array = \
                            self.position_estimator.estimate_object_positions(
                                    bbox_dict_list, pc_point_list, 
                                    cropped_pc_label='taskboard', 
                                    debug=self.debug
                        )
                        tb_points_array = object_points_dict['taskboard']

                        ## ----------------------------------------
                        ## Publishing Results:
                        ## ----------------------------------------

                        object_list_msg = ObjectList()
                        marker_array_msg = MarkerArray()

                        for label, object_position in object_positions_dict.items():
                            # Note: ROS2 geometry_msgs/Point accepts only floats, not numpy.float64, hence the cast:
                            object_msg = Object(label=label, pose=Pose(position=Point(**dict(zip(['x', 'y', 'z'], 
                                                                                                 [float(value) for value in object_position]))), 
                                                                       orientation=Quaternion(**dict(zip(['x', 'y', 'z', 'w'], (0., 0., 0., 1.))))))
                            object_msg.header.frame_id = self.current_camera_info_msg.header.frame_id
                            object_list_msg.objects.append(object_msg)

                            marker_msg, text_marker_msg = self.get_point_markers(
                                        object_position,
                                        frame_id=self.current_camera_info_msg.header.frame_id,
                                        label=object_msg.label,
                                        color_value=self.class_colors_dict[object_msg.label]
                            )
                            marker_array_msg.markers.append(marker_msg)
                            marker_array_msg.markers.append(text_marker_msg)

                        self.object_positions_publisher.publish(object_list_msg)
                        self.object_marker_publisher.publish(marker_array_msg)

                        if self.debug:
                            cropped_pc_msg = PointCloud()
                            cropped_pc_msg.header.frame_id = self.current_camera_info_msg.header.frame_id
                            cropped_pc_msg.header.stamp = self.current_pc2_msg.header.stamp

                            for point in cropped_pc_points_array:
                                # Note: ROS2 geometry_msgs/Point32 accepts only floats, not numpy.float64, hence the cast:
                                cropped_pc_msg.points.append(Point32(x=float(point[0]), y=float(point[1]), z=float(point[2])))
                            self.cropped_pc_publisher.publish(cropped_pc_msg)

                        elapsed_time = time.time() - position_estimation_start_time
                        self.get_logger().info(f'Estimated object positions in {elapsed_time:.2f}s')

                        # Optionally save results: taskboard points and object positions dict.
                        # Mostly for testing and debugging.
                        if self.save_output:
                            self.get_logger().info(f'Saving output data...')
                            if not os.path.isdir(self.output_dir_path):
                                self.get_logger().info(f'Output directory ' + \
                                                       f'{output_dir_path} does not exist! ' + \
                                                       f'Creating now...')
                                os.makedirs(self.output_dir_path)

                            with open(os.path.join(self.output_dir_path, 'pose_estimator_object_positions_dict.pkl'), 'wb') as handle:
                                pickle.dump(object_positions_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            np.save(os.path.join(self.output_dir_path, 'pose_estimator_taskboard_points.npy'), 
                                    np.stack(object_points_dict['taskboard']))


                        ## ----------------------------------------------------------------------
                        ## Orientation Estimation:
                        ## ----------------------------------------------------------------------

                        ## ----------------------------------------
                        ## Parameters and Variables:
                        ## ----------------------------------------

                        parameters = {'arrow_scale_factor': 0.2, 
                                      'plot_fitted_rectangle': True, 
                                      'plot_fitted_rectified_rectangle': True, 
                                      'hide_pc_points': False}

                        # Run orientation estimation until successful for a maximum of num_retries times.
                        for attempt_id in range(self.num_retries):
                            tb_orientation_estimation_start_time = time.time()
                            tb_tf_matrix, orientation_estimation_success, \
                                vertical_side_found, \
                                horizontal_side_found = self.position_estimator.estimate_tb_orientation(tb_points_array, 
                                                                                                       object_positions_dict, 
                                                                                                       debug=self.debug,
                                                                                                       **parameters)

                            # Compute and normalize orientation quaternion:
                            if orientation_estimation_success:
                                orientation_quaternion = np.array(quaternion_from_matrix(tb_tf_matrix))
                                orientation_quaternion /= np.linalg.norm(orientation_quaternion)
                            else:
                                self.get_logger().info(f'Could not sufficiently ' + \
                                                       f'determine taskboard orientation.')
                                self.get_logger().info(f'Will not publish ' + \
                                                       f'{self.taskboard_frame_name}.')
                                self.get_logger().info(f'Will re-attempt to estimate orientation...')
                                continue

                            elapsed_time = time.time() - tb_orientation_estimation_start_time
                            self.get_logger().info(f'Estimated taskboard orientation in {elapsed_time:.2f}s')
                            break

                        else:
                            self.get_logger().info(f'Failed to estimate TB orientation after {num_retries} attempts.')
                            self.current_detection_msg = None
                            continue

                        elapsed_time = time.time() - position_estimation_start_time
                        self.get_logger().info(f'Finished in {elapsed_time:.2f}s')
                        self.get_logger().info(f'Continuously publishing current {self.taskboard_frame_name} transform')
                    else:
                        self.get_logger().info(f'Could not get a pointcloud message from topic {pointcloud_topic}! ' + \
                                               f'Skipping pose estimation for this detection result...')
                    self.current_detection_msg = None

                    if object_list_msg is not None:
                        if orientation_quaternion is not None:
                            # Broadcast estimated taskboard frame:
                            # tb_quaternion = Quaternion(*orientation_quaternion)
                            tb_quaternion = Quaternion(**dict(zip(['x', 'y', 'z', 'w'], 
                                                                  [float(value) for value in orientation_quaternion])))

                            tf_msg = TransformStamped()
                            ## TODO: Verify ROS2 alternative:
                            tf_msg.header.stamp = self.get_clock().now().to_msg()
                            tf_msg.header.frame_id = self.current_camera_info_msg.header.frame_id
                            # tf_msg.header.frame_id = 'camera_depth_optical_frame'
                            tf_msg.child_frame_id = self.taskboard_frame_name
                            tf_msg.transform.translation = Vector3(**dict(zip(['x', 'y', 'z'], 
                                                                              [float(value) for value in object_positions_dict['taskboard']])))
                            tf_msg.transform.rotation = tb_quaternion
                            self.tf_broadcaster.sendTransform(tf_msg)

                            # TODO: Create TF to visualize taskboard_frame wrt dummy_link:
                            # tf_msg_test = TransformStamped()

                            # Re-publish objects list after adding orientations, and broadcast a frame for each:
                            updated_object_list_msg = ObjectList()

                            detected_objects_list = [object_msg.label for object_msg in object_list_msg.objects]
                            self.get_logger().info(f'[DEBUG] detected_objects_list: \n{detected_objects_list}')
                            self.get_logger().info(f'[DEBUG] self.labels_list: \n{self.labels_list}')

                            # Purge undetected objects from TF tree (RViz visualization):
                            for label in self.labels_list:
                                if label not in detected_objects_list:
                                    tf_msg = TransformStamped()
                                    tf_msg.header.stamp = self.get_clock().now().to_msg()
                                    tf_msg.header.frame_id = self.current_camera_info_msg.header.frame_id
                                    tf_msg.header.frame_id = 'non-existent'
                                    tf_msg.child_frame_id = label + '_frame'
                                    self.tf_broadcaster.sendTransform(tf_msg)

                            self.object_marker_id = 0
                            for object_msg in object_list_msg.objects:
                                label = object_msg.label
                                object_msg.pose.orientation = tb_quaternion
                                updated_object_list_msg.objects.append(object_msg)

                                tf_msg = TransformStamped()
                                ## TODO: Verify ROS2 alternative:
                                tf_msg.header.stamp = self.get_clock().now().to_msg()
                                tf_msg.header.frame_id = self.current_camera_info_msg.header.frame_id
                                # tf_msg.header.frame_id = 'camera_depth_optical_frame'
                                tf_msg.child_frame_id = label + '_frame'
                                tf_msg.transform.translation = Vector3(**dict(zip(['x', 'y', 'z'], 
                                                                                  [float(object_msg.pose.position.x), 
                                                                                   float(object_msg.pose.position.y), 
                                                                                   float(object_msg.pose.position.z)])))
                                tf_msg.transform.rotation = tb_quaternion
                                self.tf_broadcaster.sendTransform(tf_msg)

                            self.object_poses_publisher.publish(updated_object_list_msg)

                            udp_object_list_msg = updated_object_list_msg
                        else:
                            udp_object_list_msg = object_list_msg

                        # Send object pose data over UDP socket:
                        self.get_logger().info(f'Sending object pose data over over UDP...')
                        orientation_success = True if orientation_quaternion is not None else False
                        udp_message = obj_list_msg_to_json(udp_object_list_msg, 
                                                           orientation_success=orientation_success)

                        if self.debug:
                            self.get_logger().info(f'[DEBUG] UDP message: \n{udp_message}')
                        udp_message = udp_message.encode()
                        self.udp_output_socket.sendto(udp_message, (self.udp_ip, self.udp_output_port))

                # pose_estimator.rate_object.sleep()
                time.sleep(float(1. / self.rate))
        except KeyboardInterrupt:
            self.get_logger().info(f'Stopping node')
            raise SystemExit


def main(args=None):
    ## ----------------------------------------------------------------------
    ## ROS Initializations:
    ## ----------------------------------------------------------------------
    rclpy.init(args=args)
    pose_estimator_node = PoseEstimatorNode()

    ## ----------------------------------------------------------------------
    ## Estimator Execution:
    ## ----------------------------------------------------------------------

    pose_estimator_node.get_logger().info(f'Waiting for first camera info message ' + \
                                     f'on topic: {pose_estimator_node.camera_info_topic}')
    try:
        while pose_estimator_node.current_camera_info_msg is None:
            rclpy.spin_once(pose_estimator_node)
            # Will block when not spinning:
            # pose_estimator_node.rate_object.sleep()
            time.sleep(float(1. / pose_estimator_node.rate))
    # except (KeyboardInterrupt, rospy.ROSInterruptException):
    except KeyboardInterrupt:
        pose_estimator_node.get_logger().info(f'Stopping node......')
        pose_estimator_node.destroy_node()
        rclpy.shutdown()

    pose_estimator_node.get_logger().info(f'Received first camera info message')
    pose_estimator_node.position_estimator.load_camera_params(pose_estimator_node.get_camera_params_dict())
    pose_estimator_node.get_logger().info(f'Will estimate object poses for every message ' + \
                                     f'received on topic: {pose_estimator_node.detector_result_topic}')

    try:
        pose_estimator_node.run_node()
    except SystemExit:
        rclpy.logging.get_logger('rclpy').info('Stopping node...')

    pose_estimator_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
