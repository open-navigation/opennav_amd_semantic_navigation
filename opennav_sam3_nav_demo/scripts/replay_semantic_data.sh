#!/bin/bash

set -e
source /opt/ros/jazzy/setup.bash

ros2 bag play $1 -s mcap --clock
