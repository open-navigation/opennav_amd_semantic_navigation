# Copyright (C) 2026 Open Navigation LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('opennav_sam3_inference')

    default_params = PathJoinSubstitution([pkg_share, 'config', 'sam3_inference.yaml'])

    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    image_topic = LaunchConfiguration('image_topic')
    sensor_processing_pipeline = LaunchConfiguration('sensor_processing_pipeline')
    namespace = LaunchConfiguration('namespace')
    log_level = LaunchConfiguration('log_level')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Path to the SAM3 inference parameters YAML.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Whether to use simulation time.',
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/sensors/camera_0/color/image',
            description='Input image topic to remap onto ~/image.',
        ),
        DeclareLaunchArgument(
            'sensor_processing_pipeline',
            default_value='true',
            description='Whether to process sensor data for use with costmap layer.',
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
            parameters=[params_file, {'use_sim_time': use_sim_time}],
            remappings=[
                ('~/image', image_topic),
            ],  
            arguments=['--ros-args', '--log-level', log_level],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    pkg_share, 'launch', 'include', 'sensor_processing_pipeline.launch.py'])]),
            condition=IfCondition(sensor_processing_pipeline),
        )
    ])
