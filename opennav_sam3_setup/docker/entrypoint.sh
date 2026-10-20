#!/usr/bin/env bash
# Source ROS 2 (the base image sets ROS_DISTRO) and the colcon overlay (if
# built), then run the command.
set +u
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
if [[ -f /ws/install/setup.bash ]]; then
    source /ws/install/setup.bash
fi
exec "$@"
