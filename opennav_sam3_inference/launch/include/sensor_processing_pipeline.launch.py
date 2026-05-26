from launch import LaunchDescription
from launch.actions import GroupAction

from launch_ros.actions import LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode

def generate_launch_description():

    depth_pointcloud_proc = GroupAction([
        Node(
            name='depth_pointcloud_proc_container',
            package='rclcpp_components',
            executable='component_container_isolated',
            parameters=[],
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
                ),

            ],
        ),
    ])

    return LaunchDescription([depth_pointcloud_proc])