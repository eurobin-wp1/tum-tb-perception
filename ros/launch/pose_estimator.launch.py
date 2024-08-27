#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument 
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    class_colors_file_path_launch_arg = DeclareLaunchArgument(
        'class_colors_file_path', 
        default_value=PathJoinSubstitution([
            FindPackageShare(package='tum_tb_perception'), 
            'config', 'class_colors_taskboard.yaml'
        ]),
        description='TODO'
    )
    output_dir_path_launch_arg = DeclareLaunchArgument(
        'output_dir_path', 
        default_value='/tmp',
        description='TODO'
    )
    taskboard_frame_name_launch_arg = DeclareLaunchArgument(
        'taskboard_frame_name', 
        default_value='taskboard_frame',
        description='TODO'
    )
    num_retries_launch_arg = DeclareLaunchArgument(
        'num_retries', 
        default_value='3',
        description='TODO'
    )
    udp_ip_launch_arg = DeclareLaunchArgument(
        'udp_ip', 
        default_value='localhost',
        description='TODO'
    )
    udp_output_port_launch_arg = DeclareLaunchArgument(
        'udp_output_port', 
        default_value='6000',
        description='TODO'
    )
    pointcloud_topic_launch_arg = DeclareLaunchArgument(
        'pointcloud_topic', 
        default_value='/camera/camera/depth/color/points',
        description='TODO'
    )
    camera_info_topic_launch_arg = DeclareLaunchArgument(
        'camera_info_topic', 
        default_value='/camera/camera/color/camera_info',
        description='TODO'
    )
    detector_result_topic_launch_arg = DeclareLaunchArgument(
        'detector_result_topic', 
        default_value='/tum_tb_perception/detection_result',
        description='TODO'
    )
    object_positions_pub_topic_launch_arg = DeclareLaunchArgument(
        'object_positions_pub_topic', 
        default_value='/tum_tb_perception/object_positions',
        description='TODO'
    )
    object_poses_pub_topic_launch_arg = DeclareLaunchArgument(
        'object_poses_pub_topic', 
        default_value='/tum_tb_perception/object_poses',
        description='TODO'
    )
    object_marker_pub_topic_launch_arg = DeclareLaunchArgument(
        'object_marker_pub_topic', 
        default_value='/tum_tb_perception/object_markers',
        description='TODO'
    )
    cropped_pc_pub_topic_launch_arg = DeclareLaunchArgument(
        'cropped_pc_pub_topic', 
        default_value='/tum_tb_perception/cropped_pc',
        description='TODO'
    )
    save_output_launch_arg = DeclareLaunchArgument(
        'save_output', 
        default_value='False',
        description='TODO'
    )
    rate_launch_arg = DeclareLaunchArgument(
        'rate', 
        default_value='10',
        description='TODO'
    )
    debug_launch_arg = DeclareLaunchArgument(
        'debug', 
        default_value='False',
        description='TODO'
    )

    pose_estimator_node = Node(
        package='tum_tb_perception',
        namespace='tum_tb_perception',
        executable='pose_estimator_node.py',
        name='pose_estimator',
        parameters=[
            {'class_colors_file_path': LaunchConfiguration('class_colors_file_path')},
            {'output_dir_path': LaunchConfiguration('output_dir_path')},
            {'taskboard_frame_name': LaunchConfiguration('taskboard_frame_name')},
            {'num_retries': LaunchConfiguration('num_retries')},
            {'udp_ip': LaunchConfiguration('udp_ip')},
            {'udp_output_port': LaunchConfiguration('udp_output_port')},
            {'pointcloud_topic': LaunchConfiguration('pointcloud_topic')},
            {'camera_info_topic': LaunchConfiguration('camera_info_topic')},
            {'detector_result_topic': LaunchConfiguration('detector_result_topic')},
            {'object_positions_pub_topic': LaunchConfiguration('object_positions_pub_topic')},
            {'object_poses_pub_topic': LaunchConfiguration('object_poses_pub_topic')},
            {'object_marker_pub_topic': LaunchConfiguration('object_marker_pub_topic')},
            {'cropped_pc_pub_topic': LaunchConfiguration('cropped_pc_pub_topic')},
            {'save_output': LaunchConfiguration('save_output')},
            {'rate': LaunchConfiguration('rate')},
            {'debug': LaunchConfiguration('debug')},
        ],
    )

    return LaunchDescription([
        class_colors_file_path_launch_arg,
        output_dir_path_launch_arg,
        taskboard_frame_name_launch_arg,
        num_retries_launch_arg,
        udp_ip_launch_arg,
        udp_output_port_launch_arg,
        pointcloud_topic_launch_arg,
        camera_info_topic_launch_arg,
        detector_result_topic_launch_arg,
        object_positions_pub_topic_launch_arg,
        object_poses_pub_topic_launch_arg,
        object_marker_pub_topic_launch_arg,
        cropped_pc_pub_topic_launch_arg,
        save_output_launch_arg,
        rate_launch_arg,
        debug_launch_arg,
        pose_estimator_node, 
    ])
