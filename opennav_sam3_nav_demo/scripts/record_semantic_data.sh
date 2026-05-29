#!/bin/bash

set -e

ros2 bag record --max-bag-size 3000000000 -s mcap \
    /ekf/imu/data \
    /ekf/status \
    /platform/cmd_vel \
    /platform/dynamic_joint_states \
    /platform/joint_states \
    /platform/odom \
    /platform/odom/filtered \
    /robot_description \
    /sensors/camera_0/color/camera_info \
    /sensors/camera_0/color/image \
    /sensors/camera_0/depth/camera_info \
    /sensors/camera_0/depth/image \
    /tf \
    /tf_static
