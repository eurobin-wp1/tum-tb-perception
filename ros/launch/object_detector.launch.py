#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument 
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    model_weights_file_path_launch_arg = DeclareLaunchArgument(
        'model_weights_file_path', 
        default_value=PathJoinSubstitution([
            FindPackageShare(package='tum_tb_perception'), 
            'models', 
            'tb_fasterrcnn_epochs_25_batches_1_tv_ratio_07_seed_2_20240121_154144.pt'
        ]),
        description='TODO'
    )
    class_colors_file_path_launch_arg = DeclareLaunchArgument(
        'class_colors_file_path', 
        default_value=PathJoinSubstitution([
            FindPackageShare(package='tum_tb_perception'), 
            'config', 'class_colors_taskboard.yaml'
        ]),
        description='TODO'
    )
    labels_file_path_launch_arg = DeclareLaunchArgument(
        'labels_file_path', 
        default_value=PathJoinSubstitution([
            FindPackageShare(package='tum_tb_perception'), 
            'config', 'labels.txt'
        ]),
        description='TODO'
    )
    output_dir_path_launch_arg = DeclareLaunchArgument(
        'output_dir_path', 
        default_value='/tmp',
        description='TODO'
    )
    confidence_threshold_launch_arg = DeclareLaunchArgument(
        'confidence_threshold', 
        default_value='0.7',
        description='TODO'
    )
    run_on_ros_trigger_launch_arg = DeclareLaunchArgument(
        'run_on_ros_trigger', 
        default_value='True',
        description='TODO'
    )
    run_on_udp_trigger_launch_arg = DeclareLaunchArgument(
        'run_on_udp_trigger', 
        default_value='False',
        description='TODO'
    )
    udp_ip_launch_arg = DeclareLaunchArgument(
        'udp_ip', 
        default_value='localhost',
        description='TODO'
    )
    udp_trigger_port_launch_arg = DeclareLaunchArgument(
        'udp_trigger_port', 
        default_value='5000',
        description='TODO'
    )
    image_topic_launch_arg = DeclareLaunchArgument(
        'image_topic', 
        default_value='/camera/camera/color/image_raw',
        description='TODO'
    )
    trigger_topic_launch_arg = DeclareLaunchArgument(
        'trigger_topic', 
        default_value='/tum_tb_perception/detector_trigger',
        description='TODO'
    )
    image_pub_topic_launch_arg = DeclareLaunchArgument(
        'image_pub_topic', 
        default_value='/tum_tb_perception/detection_images',
        description='TODO'
    )
    input_image_pub_topic_launch_arg = DeclareLaunchArgument(
        'input_image_pub_topic', 
        default_value='/tum_tb_perception/input_images',
        description='TODO'
    )
    detection_pub_topic_launch_arg = DeclareLaunchArgument(
        'detection_pub_topic', 
        default_value='/tum_tb_perception/detection_result',
        description='TODO'
    )
    publish_visual_output_launch_arg = DeclareLaunchArgument(
        'publish_visual_output', 
        default_value='True',
        description='TODO'
    )
    save_output_launch_arg = DeclareLaunchArgument(
        'save_output', 
        default_value='False',
        description='TODO'
    )
    device_launch_arg = DeclareLaunchArgument(
        'device', 
        default_value='cpu',
        description='TODO'
    )
    rate_launch_arg = DeclareLaunchArgument(
        'rate', 
        default_value='10',
        description='TODO'
    )

    # Note: also prints a bunch of timer and subscription-related debug msgs:
    # log_level_launch_arg = DeclareLaunchArgument(
    #     'log_level', 
    #     default_value='info',
    #     description='TODO'
    # )
    debug_launch_arg = DeclareLaunchArgument(
        'debug', 
        default_value='False',
        description='TODO'
    )

    cnn_detector_node = Node(
        package='tum_tb_perception',
        namespace='tum_tb_perception',
        executable='continuous_cnn_detector_node.py',
        name='cnn_detector_node',
        parameters=[
            {'model_weights_file_path': LaunchConfiguration('model_weights_file_path')},
            {'class_colors_file_path': LaunchConfiguration('class_colors_file_path')},
            {'labels_file_path': LaunchConfiguration('labels_file_path')},
            {'output_dir_path': LaunchConfiguration('output_dir_path')},
            {'confidence_threshold': LaunchConfiguration('confidence_threshold')},
            {'run_on_ros_trigger': LaunchConfiguration('run_on_ros_trigger')},
            {'run_on_udp_trigger': LaunchConfiguration('run_on_udp_trigger')},
            {'udp_ip': LaunchConfiguration('udp_ip')},
            {'udp_trigger_port': LaunchConfiguration('udp_trigger_port')},
            {'image_topic': LaunchConfiguration('image_topic')},
            {'trigger_topic': LaunchConfiguration('trigger_topic')},
            {'image_pub_topic': LaunchConfiguration('image_pub_topic')},
            {'input_image_pub_topic': LaunchConfiguration('input_image_pub_topic')},
            {'detection_pub_topic': LaunchConfiguration('detection_pub_topic')},
            {'publish_visual_output': LaunchConfiguration('publish_visual_output')},
            {'save_output': LaunchConfiguration('save_output')},
            {'device': LaunchConfiguration('device')},
            {'rate': LaunchConfiguration('rate')},
            {'debug': LaunchConfiguration('debug')},
        ],
        # arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    )

    return LaunchDescription([
        model_weights_file_path_launch_arg,
        class_colors_file_path_launch_arg,
        labels_file_path_launch_arg,
        output_dir_path_launch_arg,
        confidence_threshold_launch_arg,
        run_on_ros_trigger_launch_arg,
        run_on_udp_trigger_launch_arg,
        udp_ip_launch_arg,
        udp_trigger_port_launch_arg,
        image_topic_launch_arg,
        trigger_topic_launch_arg,
        image_pub_topic_launch_arg,
        input_image_pub_topic_launch_arg,
        detection_pub_topic_launch_arg,
        publish_visual_output_launch_arg,
        save_output_launch_arg,
        device_launch_arg,
        rate_launch_arg,
        debug_launch_arg,
        # log_level_launch_arg,
        cnn_detector_node, 
    ])
