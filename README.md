




Note: the installation of the ROS package using `catkin` requires the `catkin_pkg` Python package. This can be installed on Linux with:
```
sudo apt-get install python3-catkin-pkg
```

Note: the recommended wayt to run the components is through the launch files, because thet already set various configurations (config file paths, etc.).


## Future Plans:

* Separate core code and ROS interfaces.
* Include complete code and instructions for training the detection model.
* Implementing continuous detection + pose estimation (detection may require GPU).
* Implementing pose estimation in C++ (if pointcloud processing run-time improves).
* Create a ROS2 interface.
