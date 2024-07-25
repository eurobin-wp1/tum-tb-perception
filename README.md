# eurobin-perception

A ROS package that contains the core perception libraries and tools for the euRobin challenge.

Designed for and tested on Ubuntu 20.04 LTS and ROS Noetic with Python 3.8.

<b>Note: The package is in an initial development phase. It is unstable and may significantly change in concept and implementation.</b>

## Contents
- [➤ Overview](#overview)
- [➤ Main Components](#main-components)
    - [Object Detector](#object-detector)
    - [Pose Estimator](#pose-estimator)
- [➤ Installation](#installation)
- [➤ Usage](#usage)
- [➤ Directory Structure](#directory-structure)
- [➤ Dependencies](#dependencies)
- [➤ Future Plans](#future-plans)

## Overview

To solve the euRobin challenge, <b>eurobin_perception</b> is intended to implement the following perception functionalities by utilizing RGB-D sensor data:
* detecting and localizing a taskboard and its components to enable a robot to reach and manipulate each component,
* estimating the positions of elements on the LCD screen to solve the slider task.

The package currently contains an implementation of the detection and localization of taskboard components, which consists of two parts. A [Faster R-CNN](https://pytorch.org/vision/main/models/faster_rcnn.html) CNN model is trained on a dataset of RGB images of the taskboard and deployed on an <b>object detector</b> node. A <b>pose estimator</b> node then uses the detection result and depth point cloud data to estimate the 3D position of each detected taskboard component and the pose of the taskboard (position and orientation).

## Main Components

### Object Detector

The `single_cnn_detector_node` runs once when executed and performs the following:
* loads a pre-trained detection model
* runs the model on a single input image from a ROS topic
* publishes and optionally visualizes the detection result

### Pose Estimator

The `pose_estimator_node` runs continuously; for each received detection result, it:
* extracts segments of the point cloud that correspond to each object that was detected in the image; for this, each 3D point is back-projected onto the image plane to determine if it lies within a given bounding box.
* estimates the 3D position of each detected object through a measure of the aggregate of its pointcloud segment (currently the mean).
* estimates the 3D orientation of the taskboard from its point cloud segment.
* publishes and optionally visualizes the object positions and a coordinate frame that describes the pose of the taskboard.

In order to estimate the taskboard orientation, we find a coordinate frame whose i) `z`-axis points perpendicular to the face and upwards, ii) `x`-axis points from the left to the right side of the board, and iii) `y`-axis points from the bottom to the top side of the board. This is achieved by:
* filtering the point cloud segment to remove outliers
* fitting a plane through the remaining points and computing its normal vector; this vector represents the `z`-axis of the taskboard
* projecting all points onto the plane and fitting a minimum-bounding rectangle on the points; the sides of this rectangle are parallel to the `x` and `y` axes of the taskboard
* if possible, finding the specific directions of the `x` and `y` vectors using information about the 3D positions of objects:
  * dividing the taskboard rectangle into four quadrants
  * recognizing quadrants 1-4 from the objects that lie within them (e.g. `slider` is expected to be inside quadrant 4)
  * finding the correspondin corners of the taskboard
  * setting the directions of `x` and `y` such that they point towards the two right-side corners and the top-side corners, respectively.

## Installation

### From Source

After cloning this repository into your catkin workspace, it is recommended to first install the Python package dependencies using `pip` by running the following within this directory:
```
pip install -r requirements.txt
```

Build the package using:
```
catkin build eurobin_perception
```

## Usage

Start the pose estimator node by running:
```bash
roslaunch eurobin_perception pose_estimator.launch
```

Execute the single object detector node by running:
```bash
roslaunch eurobin_perception object_detector_single.launch
```

Note: the recommended way to run the components is through the launch files, because they ensure the correct configuration of various parameters (config file paths, etc.).

## Directory Structure

<details>
<summary> Package Files </summary>

```
eurobin-perception
│
├── src
│   └── eurobin_perception
│       ├── __init__.py
│       ├── dataset.py
│       ├── models.py
│       ├── utils.py
│       └── visualization.py
│
├── ros
│   └── scripts
│   |   ├── single_cnn_detector_node
│   |   └── pose_estimator_node
│   └── launch
│       ├── object_detector_single.launch
│       └── pose_estimator.launch
│
├── scripts
│   └── test_dmp_approximator.py
│
├── config/
├── models/
├── msg/
├── setup.py
├── CMakeLists.txt
├── package.xml
├── requirements.txt
├── README.md
└── LICENSE
```

</details>


## Dependencies

* `cv2`
* `cv_bridge`
* `matplotlib`
* `numpy`
* `pandas`
* `rospy`
* `rospkg`
* `scipy`
* `torch`
* `torchvision`

For ROS:

* `geometry_msgs`
* `std_msgs`
* `sensor_msgs`
* `tf2_ros`
* `tf_conversions`
* `visualization_msgs`

## Future Plans

- [X] Separate core code and ROS interfaces.
- [X] Implement estimation of slider task solution.
- [ ] Include complete code and instructions for training the detection model.
- [ ] Implement continuous detection + pose estimation (detection may require GPU).
- [ ] Implement pose estimation in C++ (if pointcloud processing run-time improves).
- [ ] Create a ROS2 interface.

<!-- TODO: Add references, etc., if any
## Credits
* ...
 -->

