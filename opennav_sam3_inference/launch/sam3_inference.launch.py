# Copyright (C) 2026 Open Navigation LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import SetEnvironmentVariable


def generate_launch_description():
    pkg_share = FindPackageShare('opennav_sam3_inference')

    default_params = PathJoinSubstitution([pkg_share, 'config', 'sam3_inference.yaml'])

    params_file = LaunchConfiguration('params_file')
    image_topic = LaunchConfiguration('image_topic')
    segmentation_topic = LaunchConfiguration('segmentation_topic')
    label_mask_topic = LaunchConfiguration('label_mask_topic')
    namespace = LaunchConfiguration('namespace')
    log_level = LaunchConfiguration('log_level')

    return LaunchDescription([
        SetEnvironmentVariable(
            name='LD_LIBRARY_PATH',
            value=[
                '/opt/rocm-7.2.0/lib/libmigraphx_c.so.3:'
                '/opt/rocm-7.2.0/lib/migraphx/lib/libmigraphx.so.2016000.0:',
                EnvironmentVariable('LD_LIBRARY_PATH', default_value=''),
            ],
        ),
        SetEnvironmentVariable(
            name='PYTHONPATH',
            value=[
                '/opt/rocm-7.2.0/lib:',
                EnvironmentVariable('PYTHONPATH', default_value=''),
            ],
        ),

        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Path to the SAM3 inference parameters YAML.',
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/color/image_raw',
            description='Input image topic to remap onto ~/image.',
        ),
        DeclareLaunchArgument(
            'segmentation_topic',
            default_value='/sam3/segmentation_mask',
            description='Output segmentation mask topic to remap onto ~/segmentation_mask.',
        ),
        DeclareLaunchArgument(
            'label_mask_topic',
            default_value='/sam3/label_mask',
            description='Output label mask topic to remap onto ~/label_mask.',
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
                ('~/label_mask', label_mask_topic),
            ],
            additional_env={
                'LD_PRELOAD': '/opt/rocm-7.2.0/lib/libmigraphx_c.so.3:'
                '/opt/rocm-7.2.0/lib/migraphx/lib/libmigraphx.so.2016000.0'},
            arguments=['--ros-args', '--log-level', log_level],
        ),
    ])
