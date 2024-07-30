cmake_minimum_required(VERSION 3.0.2)
project(tum_tb_perception)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  sensor_msgs
  geometry_msgs
  message_generation
  visualization_msgs
  tf2_ros
  tf_conversions
)

catkin_python_setup()

## Generate messages in the 'msg' folder
add_message_files(
  FILES
    BoundingBox.msg
    BoundingBoxList.msg
    Object.msg
    ObjectList.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
    std_msgs
    geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    sensor_msgs
    message_runtime
)

# include_directories(
#   ${catkin_INCLUDE_DIRS}
# )
