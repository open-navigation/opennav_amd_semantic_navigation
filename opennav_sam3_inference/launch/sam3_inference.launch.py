# Copyright (C) 2026 Open Navigation LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from glob import glob

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _resolve_rocm_lib_dirs():
    """Resolve the patched ROCm 7.2 (MIGraphX 2.16) lib directories.

    The .mxr backbone artifacts are compiled with the patched MIGraphX and must
    be loaded with the matching runtime, so the node needs that runtime ahead of
    any other libmigraphx on the system (hence LD_PRELOAD). This mirrors the
    autodetection used in setup.sh and tracker.py: honour ROCM_PATH, otherwise
    fall back to the newest /opt/rocm-7.2.* install.

    Returns (lib_dir, migraphx_lib_dir).
    """
    base = os.environ.get('ROCM_PATH', '').rstrip('/')
    if not (base and os.path.isdir(os.path.join(base, 'lib'))):
        base = next(
            (p for p in sorted(glob('/opt/rocm-7.2.*'), reverse=True)
             if os.path.isdir(os.path.join(p, 'lib'))),
            '/opt/rocm-7.2.0',
        )
    lib_dir = os.path.join(base, 'lib')
    migraphx_lib_dir = os.path.join(lib_dir, 'migraphx', 'lib')
    return lib_dir, migraphx_lib_dir


def generate_launch_description():
    pkg_share = FindPackageShare('opennav_sam3_inference')

    default_params = PathJoinSubstitution([pkg_share, 'config', 'sam3_inference.yaml'])

    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    image_topic = LaunchConfiguration('image_topic')
    sensor_processing_pipeline = LaunchConfiguration('sensor_processing_pipeline')
    namespace = LaunchConfiguration('namespace')
    log_level = LaunchConfiguration('log_level')

    # Patched-MIGraphX runtime location (autodetected, ROCM_PATH-aware).
    lib_dir, migraphx_lib_dir = _resolve_rocm_lib_dirs()
    ld_preload = (
        f'{lib_dir}/libmigraphx_c.so.3:'
        f'{migraphx_lib_dir}/libmigraphx.so.2016000.0'
    )

    return LaunchDescription([
        # Prepend the patched-MIGraphX lib dirs (directories, not files) so its
        # shared objects resolve ahead of any system-default ROCm install.
        SetEnvironmentVariable(
            name='LD_LIBRARY_PATH',
            value=[
                f'{lib_dir}:{migraphx_lib_dir}:',
                EnvironmentVariable('LD_LIBRARY_PATH', default_value=''),
            ],
        ),
        # The patched MIGraphX Python binding lives in the ROCm lib dir.
        SetEnvironmentVariable(
            name='PYTHONPATH',
            value=[
                f'{lib_dir}:',
                EnvironmentVariable('PYTHONPATH', default_value=''),
            ],
        ),
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
            # LD_PRELOAD scoped to this node only: force the patched MIGraphX
            # runtime so the .mxr backbone artifacts deserialize correctly.
            additional_env={'LD_PRELOAD': ld_preload},
            arguments=['--ros-args', '--log-level', log_level],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    pkg_share, 'launch', 'include', 'sensor_processing_pipeline.launch.py'])]),
            condition=IfCondition(sensor_processing_pipeline),
        ),
    ])
