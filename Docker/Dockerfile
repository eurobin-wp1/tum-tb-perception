FROM nvidia/cuda:12.4.0-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_UID=1000
ARG USER_GID=1000
ARG USERNAME=user

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get install -y curl gnupg lsb-release
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

RUN apt-get update -y && apt-get install -y --allow-unauthenticated \
    git \
    vim \
    ranger \
    python3-pip \
    python3-catkin-tools \
    ros-noetic-ros-base \
    ros-noetic-rviz \
    ros-noetic-tf2-ros \
    ros-noetic-tf-conversions \
    wget \
    time \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/$USERNAME/workspace/catkin_ws/src
RUN git clone https://github.com/eurobin-wp1/tum-tb-perception.git --branch noetic /home/$USERNAME/workspace/catkin_ws/src/tum-tb-perception
RUN . /opt/ros/noetic/setup.sh && cd /home/$USERNAME/workspace/catkin_ws && catkin build

RUN chown -R $USER_UID:$USER_GID /home/$USERNAME/workspace

# Set the default user
USER $USERNAME

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "source /home/$USERNAME/workspace/catkin_ws/devel/setup.bash" >> ~/.bashrc
RUN pip3 install -r /home/$USERNAME/workspace/catkin_ws/src/tum-tb-perception/requirements.txt
RUN cd /home/$USERNAME/workspace/catkin_ws/src/tum-tb-perception && ./download_model.sh

# Temp: for humble tests:
# RUN git clone https://github.com/eurobin-wp1/tum-tb-perception.git --branch humble-devel /home/$USERNAME/workspace/tum-tb-perception

WORKDIR /home/$USERNAME/workspace
# ENTRYPOINT ["bash", ""]
