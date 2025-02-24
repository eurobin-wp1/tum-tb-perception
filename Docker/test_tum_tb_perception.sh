#!/bin/bash
set -e

## ==============================================================================================================
## Description   : Runs a basic test of the main pipeline of tum_tb_perception:
##                 * Starts a rosbag containing camera data
##                 * Launches the object detector
##                 * Launches the pose estimator
##                 * Sends a trigger for the object detector
##                 * Displays the outputs of the pose estimator on RViz
## Requirements  :
##                 1. A directory containing a ROS bag mounted in perception_ros_1_bags in the Docker container
##                 2. A ROS bag containing RS camera data (Image, PointCloud, CameraInfo)
## TODOs         :
##                 * Make mounted directory and ROS bag file names configurable
##
## ==============================================================================================================

pids=()

roscore &
pids+=($!)

# sleep 5s
while [[ "$(rostopic list)" != *"/rosout"* ]]; do
    sleep 0.1s
done

echo -e "[test_tum_tb_perception] [INFO]: Playing rosbag..."
rosbag play --loop --quiet /perception_ros_1_bags/fr3_tb_test_pose_5.bag &
pids+=($!)

echo -e "[test_tum_tb_perception] [INFO]: Starting RViz..."
rosrun rviz rviz -d /home/user/workspace/catkin_ws/src/tum-tb-perception/ros/config/realsense_output_config_ros1.rviz &
pids+=($!)

echo -e "[test_tum_tb_perception] [INFO]: Launching object detector..."
roslaunch tum_tb_perception object_detector.launch &
pids+=($!)

echo -e "[test_tum_tb_perception] [INFO]: Waiting for object detector to launch..."
while [[ "$(rostopic list)" != *"/camera/color/image_raw"* ]]; do
    sleep 0.1s
done
echo -e "[test_tum_tb_perception] [INFO]: Object detector has launched!"

echo -e "[test_tum_tb_perception] [INFO]: Launching pose estimator..."
roslaunch tum_tb_perception pose_estimator.launch &
pids+=($!)

echo -e "[test_tum_tb_perception] [INFO]: Waiting for pose estimator to launch..."
while [[ "$(rostopic list)" != *"/tum_tb_perception/object_poses"* ]]; do
    sleep 0.1s
done
echo -e "[test_tum_tb_perception] [INFO]: Pose estimator has launched!"

echo -e "[test_tum_tb_perception] [INFO]: Triggering object detection..."
rostopic pub -1 /tum_tb_perception/detector_trigger std_msgs/Bool "data: true"

sleep 5s

echo -e "[test_tum_tb_perception] [INFO]: Press CTRL+C to end..."
( trap exit SIGINT ; read -r -d '' _ </dev/tty ) ## wait for Ctrl-C

echo -e "[test_tum_tb_perception] [INFO]: Killing all processes"
## Kill all started processes:
for pid in ${pids[@]}; do
    # Send SIGINT signal:
    kill -INT $pid
done
