cmake_minimum_required(VERSION 3.8)
project(tum_tb_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
## Needed, but source version does not implement cmake files to find package:
## Note: also requires pip install transforms3d.
# find_package(tf_transformations REQUIRED)
## Needed, but source version does not implement cmake files to find package:
# find_package(sensor_msgs_py REQUIRED)

## Not available:
# find_package(message_runtime REQUIRED)
# find_package(tf_conversions REQUIRED)

find_package(tum_tb_perception_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

## Lead to unresolvable error, so msgs are imported from separate package tum_tb_perception_msgs:
# find_package(rosidl_default_generators REQUIRED)
# rosidl_generate_interfaces(${PROJECT_NAME}
#   "msg/Object.msg"
#   "msg/ObjectList.msg"
#   "msg/BoundingBox.msg"
#   "msg/BoundingBoxList.msg"
#   DEPENDENCIES geometry_msgs
# )
# rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")
# target_link_libraries(${library_name} "${cpp_typesupport_target}")

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  ros/${PROJECT_NAME}/continuous_cnn_detector_node.py
  ros/${PROJECT_NAME}/pose_estimator_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  ros/launch
  models
  config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
