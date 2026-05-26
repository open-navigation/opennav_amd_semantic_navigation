#!/bin/bash

set -e

source /opt/ros/jazzy/setup.bash
source ~/amd_ws/install/setup.bash

# ros2 launch opennav_sam3_inference sam3_inference.launch.py &
# ros2 launch opennav_sam3_nav_demo nav2.launch.py &

ros2 bag play $1 -s mcap --clock