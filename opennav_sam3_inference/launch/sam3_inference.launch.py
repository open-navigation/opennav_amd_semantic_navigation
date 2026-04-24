# Copyright (C) 2026 Open Navigation LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('opennav_sam3_inference')

    default_params = PathJoinSubstitution([pkg_share, 'config', 'sam3_inference.yaml'])

    params_file = LaunchConfiguration('params_file')
    image_topic = LaunchConfiguration('image_topic')
    segmentation_topic = LaunchConfiguration('segmentation_topic')
    namespace = LaunchConfiguration('namespace')
    log_level = LaunchConfiguration('log_level')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Path to the SAM3 inference parameters YAML.',
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/image_raw',
            description='Input image topic to remap onto ~/image.',
        ),
        DeclareLaunchArgument(
            'segmentation_topic',
            default_value='/sam3/segmentation_mask',
            description='Output topic to remap onto ~/segmentation_mask.',
        ),
        DeclareLaunchArgument(
            'namespace',
            default_value='',
            description='Optional namespace for the node.',
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='ros2 log level (debug, info, warn, error, fatal).',
        ),
        Node(
            package='opennav_sam3_inference',
            executable='sam3_node',
            name='sam3_inference',
            namespace=namespace,
            output='screen',
            emulate_tty=True,
            parameters=[params_file],
            remappings=[
                ('~/image', image_topic),
                ('~/segmentation_mask', segmentation_topic),
            ],
            arguments=['--ros-args', '--log-level', log_level],
        ),
    ])
