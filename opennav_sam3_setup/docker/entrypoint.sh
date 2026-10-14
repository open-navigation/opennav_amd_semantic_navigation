#!/usr/bin/env bash
# Source ROS 2 Jazzy and the colcon overlay (if built), then run the command.
set +u
source /opt/ros/jazzy/setup.bash
if [[ -f /ws/install/setup.bash ]]; then
    source /ws/install/setup.bash
fi
exec "$@"
