#!/usr/bin/env python3

"""
Runs a CNN detector continuously on every incoming image messages and publishes
the results.
Optionally runs the detector only after a ROS/UDP trigger message.

"""


import os
import sys
import time
import json
import signal
import pickle
import socket
import datetime

import cv2
import rclpy
import torch

from rclpy import Parameter
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from tum_tb_perception_msgs.msg import BoundingBox, BoundingBoxList

from tum_tb_perception.image_detection import ImageDetector

# Test: get pkg path for arg default values:
from launch_ros.substitutions import FindPackageShare

supported_torch_devices_ = ['cpu', 'gpu']

## ----------------------------------------------------------------------
## UDP Parameters
## ----------------------------------------------------------------------

udp_buffer_size_ = 1024

## ----------------------------------------------------------------------
## ROS Nodes, Callbacks and Message Initializations:
## ----------------------------------------------------------------------

class CNNDetectorNode(Node):

    def __init__(self):
        super().__init__('cnn_detector')

        self.pkg_share_path = FindPackageShare(package='tum_tb_perception').find('tum_tb_perception')

        # Get node parameters:
        self.declare_parameter('model_weights_file_path', 
            os.path.join(self.pkg_share_path, 'models/tb_fasterrcnn_epochs_25_batches_1_tv_ratio_07_seed_2_20240121_154144.pt'))
        self.declare_parameter('class_colors_file_path', 
            os.path.join(self.pkg_share_path, 'config/class_colors_taskboard.yaml'))
        self.declare_parameter('labels_file_path', 'config/labels.txt')
        self.declare_parameter('output_dir_path', '/tmp')
        self.declare_parameter('confidence_threshold', 0.7)
        self.declare_parameter('run_on_ros_trigger', True)
        self.declare_parameter('run_on_udp_trigger', False)
        self.declare_parameter('udp_ip', 'localhost')
        self.declare_parameter('udp_trigger_port', 5000)
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('trigger_topic', '/tum_tb_perception/detector_trigger')
        self.declare_parameter('image_pub_topic', '/tum_tb_perception/detection_images')
        self.declare_parameter('input_image_pub_topic', '/tum_tb_perception/input_images')
        self.declare_parameter('detection_pub_topic', '/tum_tb_perception/detection_result')
        self.declare_parameter('publish_visual_output', True)
        self.declare_parameter('save_output', False)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('rate', 10)
        self.declare_parameter('debug', False)

        self.model_weights_file_path = self.get_parameter('model_weights_file_path').value
        self.class_colors_file_path = self.get_parameter('class_colors_file_path').value
        self.labels_file_path = self.get_parameter('labels_file_path').value
        self.output_dir_path = self.get_parameter('output_dir_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.run_on_ros_trigger = self.get_parameter('run_on_ros_trigger').value
        self.run_on_udp_trigger = self.get_parameter('run_on_udp_trigger').value
        self.udp_ip = self.get_parameter('udp_ip').value
        self.udp_trigger_port = self.get_parameter('udp_trigger_port').value
        self.image_topic = self.get_parameter('image_topic').value
        self.trigger_topic = self.get_parameter('trigger_topic').value
        self.image_pub_topic = self.get_parameter('image_pub_topic').value
        self.input_image_pub_topic = self.get_parameter('input_image_pub_topic').value
        self.detection_pub_topic = self.get_parameter('detection_pub_topic').value
        self.publish_visual_output = self.get_parameter('publish_visual_output').value
        self.save_output = self.get_parameter('save_output').value
        self.device = self.get_parameter('device').value
        self.rate = self.get_parameter('rate').value
        self.debug = self.get_parameter('debug').value

        # Initialize subscribers:
        self.image_subscription = self.create_subscription(Image,
                                                           self.image_topic,
                                                           self.image_callback,
                                                           10)
        if self.run_on_ros_trigger:
            self.trigger_subscription = self.create_subscription(Bool,
                                                                 self.trigger_topic,
                                                                 self.trigger_callback,
                                                                 10)

        # Initialize publishers:
        self.bb_publisher = self.create_publisher(BoundingBoxList, self.detection_pub_topic, 10)
        if self.publish_visual_output:
            self.detection_image_publisher = self.create_publisher(Image, self.image_pub_topic, 10)
            self.input_image_publisher = self.create_publisher(Image, self.input_image_pub_topic, 10)

        # Initialize messages:
        self.current_image_msg = None

        # Initialize timer:
        self.timer = self.create_timer(float(1. / self.rate), self.timer_callback)
        
        # Initialize data variables:
        self.initialize()

        # self.get_logger().info(f'Waiting for trigger message...\n')

    def initialize(self):
        self.ros_triggered = False
        self.bridge = CvBridge()

        # Verify device for detection model:
        if self.device in supported_torch_devices_:
            self.get_logger().info(f'Will run model on {self.device}')
        else:
            self.get_logger().info(f'Invalid value for device parameter! Must be one of {supported_torch_devices_}')
            raise SystemExit
        if self.device == 'gpu' and not torch.cuda.is_available():
            self.get_logger().info(f'Could not detect gpu device! Terminating.')
            raise SystemExit

        # Verify trigger setting (ROS OR UDP):
        if self.run_on_ros_trigger and self.run_on_udp_trigger:
            self.get_logger().info(f'Node supports trigger messages from' + \
                                   ' either ROS or UDP, but run_on_ros_trigger and' + \
                                   ' run_on_udp_trigger were both set to true!')
            self.get_logger().info(f'Terminating.')
            raise SystemExit

        # Set up output data directory:
        if self.save_output:
            output_sub_dir_path = 'cnn_detector_output_' + \
                                  datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir_path = os.path.join(self.output_dir_path, output_sub_dir_path)
            self.get_logger().info(f'Saving output data in ' + \
                                   f'{output_dir_path}')

            if not os.path.isdir(output_dir_path):
                self.get_logger().info(f'Output directory does not exist! ' + \
                                       f'Creating now...')
                os.makedirs(output_dir_path)

        ## ----------------------------------------------------------------------
        ## Detector Initialization:
        ## ----------------------------------------------------------------------

        self.get_logger().info(f'Initializing ImageDetector...')
        # self.detector = ImageDetector(labels_file_path=self.labels_file_path, 
        #                               model_weights_file_path=self.model_weights_file_path, 
        #                               class_colors_file_path=self.class_colors_file_path, 
        #                               confidence_threshold=self.confidence_threshold,
        #                               device=self.device)
        self.detector = ImageDetector(model_weights_file_path=self.model_weights_file_path, 
                                      labels_file_path=self.labels_file_path,
                                      class_colors_file_path=self.class_colors_file_path, 
                                      confidence_threshold=self.confidence_threshold,
                                      device=self.device)

        self.current_time = time.time()
        self.timer_counter = 0.

        self.rate_object = self.create_rate(self.rate)

    def image_callback(self, msg):
        self.current_image_msg = msg

        if cnn_detector.debug:
            self.get_logger().info(f'[DEBUG] Received image message')
            self.get_logger().info(f'[DEBUG] {self.current_image_msg.height}')
            self.get_logger().info(f'[DEBUG] {self.current_image_msg.width}')
            self.get_logger().info(f'[DEBUG] {self.current_image_msg.header.frame_id}')

    def trigger_callback(self, msg):
        self.get_logger().info('Received trigger ROS message')
        self.ros_triggered = msg.data

    def timer_callback(self):
        pass

    def save_data(self):
        # TODO: Move data saving here
        raise NotImplementedError

    def run_node(self):
        # raise NotImplementedError
        try:
            while rclpy.ok():
                # cnn_detector.rate_object.sleep

                if self.run_on_ros_trigger:
                    # Note: using a signal SIGINT handler here because a try-except
                    # block does not work for the inner while loop (does not terminate cleanly).
                    signal.signal(signal.SIGINT, signal.SIG_DFL);
                    while not self.ros_triggered:
                        # rate.sleep()
                        cnn_detector.rate_object.sleep()
                    self.ros_triggered = False
                elif self.run_on_udp_trigger:
                    # Note: the following will block until a message is received:
                    signal.signal(signal.SIGINT, signal.SIG_DFL);
                    udp_msg, udp_addr = udp_trigger_socket.recvfrom(udp_buffer_size_)

                    self.get_logger().info('Received trigger UDP message')
                    udp_msg_data = json.loads(udp_msg.decode())

                    try:
                        if type(udp_msg_data['trigger']).__name__ != 'str' or \
                                udp_msg_data['trigger'] != "True":
                            self.get_logger().info(f'Received invalid value ' + \
                                                   f'in UDP message dict for key trigger: ' + \
                                                   f'{udp_msg_data["trigger"]}. Will only ' + \
                                                   f'trigger on "True". Ignoring...')
                            continue
                    except Exception as e:
                        self.get_logger().info(f'Could not access trigger ' + \
                                               f'information in UDP message! Please check ' + \
                                               f'message format!. Ignoring...')
                        continue
                else:
                    # Note: probably does not exist in ROS2:
                    # try:
                    #     rate.sleep()
                    # except rospy.ROSTimeMovedBackwardsException as e:
                    #     rospy.logwarn('[cnn_detector] Caught ROSTimeMovedBackwardsException ' + \
                    #                   'when executing rate.sleep(). This can happen when ' + \
                    #                   'incoming messages had stopped, and have just ' + \
                    #                   'resumed publishing.')

                    cnn_detector.rate_object.sleep()

                self.get_logger().info('Running detection model on image...')
                detection_start_time = time.time()

                try:
                    image_cv = bridge.imgmsg_to_cv2(self.current_image_msg, "bgr8")
                except CvBridgeError as e:
                    self.get_logger().info(f'Failed to convert image message to ' + \
                                           f'opencv format! Skipping...')
                    self.get_logger().info(f'Error: {e}')
                    continue

                detector_result = selfdetector.detect_objects(
                        image_cv, 
                        return_annotated_image=self.publish_visual_output or \
                                               self.save_output
                )
                bboxes = detector_result[0]
                model_inference_time = detector_result[1]
                detection_image_cv = detector_result[2]
                self.get_logger().info(f'Model inference time: {model_inference_time:.2f}s')

                # Publish results in a BoundingBoxList message:
                bbox_list_msg = BoundingBoxList()
                for bbox_dict in bboxes:
                    bbox_msg = BoundingBox()
                    bbox_msg.xmin = bbox_dict['xmin']
                    bbox_msg.xmax = bbox_dict['xmax']
                    bbox_msg.ymin = bbox_dict['ymin']
                    bbox_msg.ymax = bbox_dict['ymax']
                    bbox_msg.label = bbox_dict['class']
                    bbox_msg.confidence = bbox_dict['confidence']

                    bbox_list_msg.bounding_boxes.append(bbox_msg)

                bb_publisher.publish(bbox_list_msg)

                if publish_visual_output:
                    input_image_msg = bridge.cv2_to_imgmsg(image_cv, encoding="bgr8")
                    ## TODO: Verify ROS2 alternative:
                    input_image_msg.header.stamp = self.get_clock().now().to_msg()
                    input_image_msg.header.frame_id = self.current_image_msg.header.frame_id
                    self.input_image_publisher.publish(input_image_msg)

                    detection_image_msg = bridge.cv2_to_imgmsg(detection_image_cv, 
                                                               encoding="bgr8")
                    ## TODO: Verify ROS2 alternative:
                    detection_image_msg.header.stamp = self.get_clock().now().to_msg()
                    detection_image_msg.header.frame_id = current_image_msg_.header.frame_id
                    self.detection_image_publisher.publish(detection_image_msg)

                detection_time = time.time() - detection_start_time
                self.get_logger().info(f'Finished in {detection_time:.2f}s')

                # Optionally save results: input image, annotated images, and detection
                # result (bboxes) in a pickle file.
                if self.save_output:
                    detection_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    with open(os.path.join(self.output_dir_path, 
                                           f'detection_result_{detection_str}.pkl'), 
                              'wb') as handle:
                        pickle.dump(bboxes, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    cv2.imwrite(os.path.join(self.output_dir_path, 
                                             f'input_image_{detection_str}.png'), 
                                image_cv)
                    cv2.imwrite(os.path.join(self.output_dir_path, 
                                             f'detection_image_{detection_str}.png'), 
                                detection_image_cv)

        except KeyboardInterrupt:
            self.get_logger().info(f'Stopping node')
            raise SystemExit


def main(args=None):
    ## ----------------------------------------------------------------------
    ## ROS Initializations:
    ## ----------------------------------------------------------------------
    rclpy.init(args=args)
    cnn_detector = CNNDetectorNode()

    # Test: read launch arg value:
    # Note: avoided, since will print with alot of other ROS-specific debug msgs:
    # cnn_detector.get_logger().debug(f'Loading class_colors_file_path from ' \
    #                                 f'{cnn_detector.class_colors_file_path}')
    if cnn_detector.debug:
        cnn_detector.get_logger().info(f'Loading model from ' \
                                       f'{cnn_detector.model_weights_file_path}')
        cnn_detector.get_logger().info(f'[DEBUG] confidence_threshold: ' \
                                       f'{cnn_detector.confidence_threshold}')

    ## ----------------------------------------------------------------------
    ## Detector Execution:
    ## ----------------------------------------------------------------------

    cnn_detector.get_logger().info(f'Subscribing to image topic: ' \
                                   f'{cnn_detector.image_topic}')
    cnn_detector.get_logger().info(f'Waiting for reception of first image message...')

    # cnn_detector.get_logger().info(f'[DEBUG] cnn_detector.get_clock().now(): {cnn_detector.get_clock().now()}')
    # cnn_detector.get_logger().info(f'[DEBUG] type(cnn_detector.get_clock().now()): {type(cnn_detector.get_clock().now())}')
    # cnn_detector.get_logger().info(f'[DEBUG] cnn_detector.get_clock().now().to_msg(): {cnn_detector.get_clock().now().to_msg()}')
    # cnn_detector.get_logger().info(f'[DEBUG] type(cnn_detector.get_clock().now().to_msg()): {type(cnn_detector.get_clock().now().to_msg())}')
    # test_msg = Image()
    # test_msg.header.stamp = cnn_detector.get_clock().now().to_msg()

    try:
        while cnn_detector.current_image_msg is None:
            # rospy.sleep(0.1)
            # rclpy.spin_once(cnn_detector)
            cnn_detector.rate_object.sleep()
    # except (KeyboardInterrupt, rospy.ROSInterruptException):
    except KeyboardInterrupt:
        cnn_detector.get_logger().info(f'Terminating...')
        cnn_detector.destroy_node()
        rclpy.shutdown()

    cnn_detector.get_logger().info(f'Received first image message')

    if run_on_ros_trigger:
        self.get_logger().info(f'Will run detection on the latest ' + \
                               f'image message at every trigger ROS message on ' + \
                               f'on topic {cnn_detector.trigger_topic}...')
    elif run_on_udp_trigger:
        self.get_logger().info(f'Will run detection on the latest ' + \
                               f'image message at every trigger UDP message on ' + \
                               f' over IP {udp_ip} and port {cnn_detector.udp_trigger_port}...')

        udp_trigger_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_trigger_socket.settimeout(None)
        udp_trigger_socket.bind((cnn_detector.udp_ip, cnn_detector.udp_trigger_port))
    else:
        self.get_logger().info(f'Continuously running detection on ' + \
                               f'incoming image messages...')

    # # try-except inspired by (https://answers.ros.org/question/406469/ros-2-how-to-quit-a-node-from-within-a-callback/)
    # try:
    #     rclpy.spin(cnn_detector)
    # except SystemExit:
    #     rclpy.logging.get_logger('rclpy').info('Stopping node...')

    ## TODO: test and iterate:
    try:
        cnn_detector.run_node()
    except SystemExit:
        rclpy.logging.get_logger('rclpy').info('Stopping node...')

    cnn_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()