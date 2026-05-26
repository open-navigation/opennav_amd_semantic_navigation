#!/bin/bash

set -e

source /opt/ros/jazzy/setup.bash
source ~/amd_ws/install/setup.bash

ros2 bag play $1 -s mcap --clock
