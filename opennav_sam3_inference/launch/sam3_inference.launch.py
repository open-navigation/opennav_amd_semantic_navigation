# Copyright (C) 2026 Open Navigation LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


_DEFAULT_MIGRAPHX_ROOT = '/opt/opennav-sam3/runtime/0.2.0-rc4/migraphx'
_DEFAULT_ROCM_ROOT = '/opt/rocm'


def _prepend_paths(entries, existing):
    return ':'.join([*entries, existing] if existing else entries)


def _sam3_runtime_environment():
    """Build the isolated environment for the SAM3 inference process."""
    migraphx_root = os.environ.get('SAM3_MIGRAPHX_ROOT', _DEFAULT_MIGRAPHX_ROOT)
    rocm_root = os.environ.get('SAM3_ROCM_PATH', _DEFAULT_ROCM_ROOT)
    return {
        'SAM3_MIGRAPHX_ROOT': migraphx_root,
        'ROCM_PATH': rocm_root,
        'PATH': _prepend_paths(
            [f'{migraphx_root}/bin', f'{rocm_root}/bin'],
            os.environ.get('PATH', ''),
        ),
        'PYTHONPATH': _prepend_paths(
            [f'{migraphx_root}/lib'], os.environ.get('PYTHONPATH', '')
        ),
        'LD_LIBRARY_PATH': _prepend_paths(
            [
                f'{migraphx_root}/lib',
                f'{migraphx_root}/lib/migraphx/lib',
                f'{rocm_root}/lib',
                f'{rocm_root}/lib64',
                f'{rocm_root}/core-7.14/lib',
                f'{rocm_root}/core-7.14/lib/host-math/lib',
                f'{rocm_root}/core-7.14/lib/rocm_sysdeps/lib',
            ],
            os.environ.get('LD_LIBRARY_PATH', ''),
        ),
        'HSA_OVERRIDE_GFX_VERSION': os.environ.get(
            'HSA_OVERRIDE_GFX_VERSION', '11.5.1'
        ),
        'MIGRAPHX_GPU_HIP_FLAGS': os.environ.get(
            'MIGRAPHX_GPU_HIP_FLAGS',
            '-Wno-error -Wno-lifetime-safety-intra-tu-suggestions',
        ),
        'TRANSFORMERS_OFFLINE': os.environ.get('TRANSFORMERS_OFFLINE', '1'),
    }


def generate_launch_description():
    pkg_share = FindPackageShare('opennav_sam3_inference')

    default_params = PathJoinSubstitution([pkg_share, 'config', 'sam3_inference.yaml'])

    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    image_topic = LaunchConfiguration('image_topic')
    sensor_processing_pipeline = LaunchConfiguration('sensor_processing_pipeline')
    namespace = LaunchConfiguration('namespace')
    log_level = LaunchConfiguration('log_level')
    runtime_environment = _sam3_runtime_environment()

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
            additional_env=runtime_environment,
            arguments=['--ros-args', '--log-level', log_level],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    pkg_share, 'launch', 'include', 'sensor_processing_pipeline.launch.py'])]),
            condition=IfCondition(sensor_processing_pipeline),
        )
    ])
