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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction

from launch_ros.actions import LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time')

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Whether to use simulation time.',
    )

    depth_pointcloud_proc = GroupAction([
        Node(
            name='depth_pointcloud_proc_container',
            package='rclcpp_components',
            executable='component_container_isolated',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=['--ros-args', '--log-level', 'info'],
            output='screen'),
        
        LoadComposableNodes(
            target_container='depth_pointcloud_proc_container',
            composable_node_descriptions =[

                ComposableNode(
                    package='image_proc',
                    plugin='image_proc::ResizeNode',
                    name='depth_image_resize',
                    remappings=[
                        ('/image/image_raw',    '/sensors/camera_0/depth/image'),
                        ('/resize/image_raw',  '/sensors/camera_0/depth_resized/image'),
                    ],
                    parameters=[{
                        'use_scale':     False,
                        'width':         424,
                        'height':        240,
                        'interpolation': 0,
                        'always_subscribe': True,
                        'use_sim_time': use_sim_time,
                    }],
                ),

                ComposableNode(
                    package='image_proc',
                    plugin='image_proc::ResizeNode',
                    name='color_image_resize',
                    remappings=[
                        ('/image/image_raw',    '/sensors/camera_0/color/image'),
                        ('/resize/image_raw',  '/sensors/camera_0/color_resized/image'),
                    ],
                    parameters=[{
                        'use_scale':     False,
                        'width':         424,
                        'height':        240,
                        'interpolation': 0,
                        'always_subscribe': True,
                        'use_sim_time': use_sim_time,
                    }],
                ),

                ComposableNode(
                    package='image_proc',
                    plugin='image_proc::ResizeNode',
                    name='label_mask_resize',
                    remappings=[
                        ('/image/image_raw',    '/sam3_inference/label_mask'),
                        ('/sam3_inference/camera_info',    '/sensors/camera_0/color/camera_info'),
                        ('/resize/image_raw',   '/sam3_inference/resized/label_mask'),
                    ],
                    parameters=[{
                        'use_scale':     False,
                        'width':         424,
                        'height':        240,
                        'interpolation': 0,  # NEAREST — required for label masks
                        'always_subscribe': True,
                        'use_sim_time': use_sim_time,
                    }],
                ),

                # If your camera provides a registered pointcloud which can be decimated
                # to a useful size for the semantic segmentation costmap layer (i.e. 320x180, 424x240, etc)
                # then the following 2 components and depth_image_resize may be removed.
                # The remaining nodes are to reduce the labeled mask size to align with the pointcloud
                # representing a useful resolution to process on an occupancy grid.
                # For an Orbecc camera, you can do this with enabling pointcloud, ordered pointcloud, depth registration
                # enabling decimation filter, and finally setting the pointcloud decimation filter factor to 4 (for 320x180)
                ComposableNode(
                    package='depth_image_proc',
                    plugin='depth_image_proc::RegisterNode',
                    name='depth_register',
                    namespace='sensors',
                    remappings=[
                        ('depth/image_rect',             '/sensors/camera_0/depth_resized/image'),
                        ('depth/camera_info',            '/sensors/camera_0/depth_resized/camera_info'),
                        ('rgb/camera_info',              '/sensors/camera_0/color_resized/camera_info'),
                        ('depth_registered/image_rect',  '/sensors/camera_0/depth_registered/image'),
                        ('depth_registered/camera_info', '/sensors/camera_0/depth_registered/camera_info'),
                    ],
                    parameters=[{
                        'use_sim_time': use_sim_time,
                    }],
                ),

                ComposableNode(
                    package='depth_image_proc',
                    plugin='depth_image_proc::PointCloudXyzNode',
                    name='point_cloud_xyz',
                    namespace='sensors',
                    remappings=[
                        ('image_rect',  '/sensors/camera_0/depth_registered/image'),
                        ('camera_info', '/sensors/camera_0/depth_registered/camera_info'),
                        ('points',      '/sensors/camera_0/points_registered'),
                    ],
                    parameters=[{
                        'use_sim_time': use_sim_time,
                    }],
                ),

            ],
        ),
    ])

    return LaunchDescription([use_sim_time_arg, depth_pointcloud_proc])