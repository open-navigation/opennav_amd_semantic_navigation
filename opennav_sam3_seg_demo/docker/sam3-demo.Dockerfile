# Copyright (C) 2026 Open Navigation LLC. All rights reserved.

FROM ubuntu:noble AS base

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8

SHELL ["/bin/bash", "-c"]

# Add a non-root user
ARG USERNAME=sam3-demo-usr
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ARG RENDER_GID=110

ENV USER=$USERNAME
ENV HOME=/home/$USERNAME

RUN userdel -f -r ubuntu \
  && groupadd --gid "${USER_GID}" "${USERNAME}" \
  && groupadd --gid "${RENDER_GID}" render \
  && useradd --uid "${USER_UID}" --gid "${USER_GID}" -m "${USERNAME}" \
  && usermod -aG video,render "${USERNAME}" \
  && apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/"${USERNAME}" \
  && chmod 0440 /etc/sudoers.d/"${USERNAME}"

# Base tooling, locale, universe repo for ROS apt sources + Install ROS 2 Jazzy
ENV ROS_DISTRO=jazzy
RUN apt-get update && apt-get install -y \
    git curl wget gnupg2 lsb-release locales software-properties-common vim \
    && locale-gen en_US.UTF-8 \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
      > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y \
        ros-${ROS_DISTRO}-ros-base \
        ros-${ROS_DISTRO}-cv-bridge \
        ros-${ROS_DISTRO}-vision-msgs \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
        python3-colcon-common-extensions \
        python3-rosdep \
    && rm -rf /var/lib/apt/lists/* \
    && printf "\nif [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then source "/opt/ros/${ROS_DISTRO}/setup.bash"; fi\n" >> ${HOME}/.bashrc

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV QT_X11_NO_MITSHM=1

USER $USERNAME

# Setup SAM3 environment
COPY --chown=${USERNAME}:${USERNAME} ./opennav_sam3_setup/setup.sh ${HOME}/opennav_sam3_setup/setup.sh
COPY --chown=${USERNAME}:${USERNAME} ./opennav_sam3_setup/requirements.txt ${HOME}/opennav_sam3_setup/requirements.txt

RUN wget -O ${HOME}/Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    && bash ${HOME}/Miniforge3.sh -b -p "${HOME}/miniforge3" \
    && printf "\nif [ -f ${HOME}/miniforge3/etc/profile.d/conda.sh ]; then source ${HOME}/miniforge3/etc/profile.d/conda.sh; fi" >> ${HOME}/.bashrc \
    && rm ${HOME}/Miniforge3.sh \
    && (cd ${HOME}/opennav_sam3_setup && ./setup.sh --skip-weights --yes) \
    && rm -r ${HOME}/opennav_sam3_setup

# Install package ROS dependencies + Camera driver build and install
COPY --chown=${USERNAME}:${USERNAME} ./opennav_sam3_inference/package.xml ${HOME}/ros2_ws/src/opennav_sam3_inference/package.xml
COPY --chown=${USERNAME}:${USERNAME} ./opennav_sam3_msgs/package.xml ${HOME}/ros2_ws/src/opennav_sam3_msgs/package.xml
COPY --chown=${USERNAME}:${USERNAME} ./opennav_sam3_seg_demo/package.xml ${HOME}/ros2_ws/src/opennav_sam3_seg_demo/package.xml

RUN git clone https://github.com/orbbec/OrbbecSDK_ROS2.git ${HOME}/ros2_ws/src/OrbbecSDK_ROS2 \
    && sudo rosdep init && rosdep update \ 
    && (cd ${HOME}/ros2_ws && sudo apt update && rosdep install --from-paths src -y --ignore-src) \
    && sudo apt install -y \
        ros-${ROS_DISTRO}-image-transport \
        ros-${ROS_DISTRO}-image-publisher \
        ros-${ROS_DISTRO}-camera-info-manager \
        usbutils \
    && rm -r ${HOME}/ros2_ws/src/opennav* \
    && mkdir -p ${HOME}/orbbec-driver \
    && source /opt/ros/${ROS_DISTRO}/setup.bash \
    && (cd ${HOME}/ros2_ws && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --install-base ${HOME}/orbbec-driver) \
    && printf "\nif [ -f ${HOME}/orbbec-driver/setup.bash ]; then source ${HOME}/orbbec-driver/setup.bash; fi\n" >> ${HOME}/.bashrc \
    && rm -r ${HOME}/ros2_ws/*

FROM base AS build

# Build and install SAM3 node interfaces
COPY --chown=${USERNAME}:${USERNAME} ./opennav_sam3_msgs ${HOME}/ros2_ws/src/opennav_sam3_msgs
RUN mkdir -p ${HOME}/opennav_sam3_msgs \
    && source /opt/ros/${ROS_DISTRO}/setup.bash \
    && (cd ${HOME}/ros2_ws && colcon build --install-base ${HOME}/opennav_sam3_msgs) \
    && rm -r ${HOME}/ros2_ws/*

# Build and install SAM3 node
COPY --chown=${USERNAME}:${USERNAME} ./opennav_sam3_inference ${HOME}/ros2_ws/src/opennav_sam3_inference
RUN mkdir -p ${HOME}/opennav_sam3_inference \
    && source /opt/ros/${ROS_DISTRO}/setup.bash \
    && source ${HOME}/miniforge3/etc/profile.d/conda.sh \
    && source ${HOME}/opennav_sam3_msgs/setup.bash \
    && (cd ${HOME}/ros2_ws \
    	&& conda activate opennav-sam3-inference \
    	&& colcon build --install-base ${HOME}/opennav_sam3_inference \
    	&& conda deactivate) \
    && rm -r ${HOME}/ros2_ws/*

# Build and install demo package
COPY --chown=${USERNAME}:${USERNAME} ./opennav_sam3_seg_demo ${HOME}/ros2_ws/src/opennav_sam3_seg_demo
RUN mkdir -p ${HOME}/opennav_sam3_seg_demo \
    && source /opt/ros/${ROS_DISTRO}/setup.bash \
    && source ${HOME}/opennav_sam3_msgs/setup.bash \
    && source ${HOME}/opennav_sam3_inference/setup.bash \
    && (cd ${HOME}/ros2_ws && colcon build --install-base ${HOME}/opennav_sam3_seg_demo) \
    && rm -r ${HOME}/ros2_ws/*

COPY --chown=${USERNAME}:${USERNAME} ./opennav_sam3_seg_demo/docker/entrypoint.sh ${HOME}/entrypoint.sh
