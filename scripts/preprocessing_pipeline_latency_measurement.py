#!/usr/bin/env python3
import argparse

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image, PointCloud2


class SyncLatencyMonitor(Node):
    def __init__(self, image_topic, pointcloud_topic, slop):
        super().__init__('sync_latency_monitor')
        self.image_sub = Subscriber(self, Image, image_topic)
        self.pc_sub = Subscriber(self, PointCloud2, pointcloud_topic)
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.pc_sub], queue_size=10, slop=slop
        )
        self.sync.registerCallback(self.callback)
        self.get_logger().info(
            f'Listening on "{image_topic}" and "{pointcloud_topic}" '
            f'with slop={slop}s'
        )

    def callback(self, image_msg, pc_msg):
        now = self.get_clock().now()
        t_image = Time.from_msg(image_msg.header.stamp)
        t_pc = Time.from_msg(pc_msg.header.stamp)

        diff_ns = abs((t_image - t_pc).nanoseconds)
        oldest = min(t_image, t_pc)
        latency_ns = (now - oldest).nanoseconds

        self.get_logger().info(
            f'Image-PC diff: {diff_ns / 1e6:.2f} ms | '
            f'Oldest->now latency: {latency_ns / 1e6:.2f} ms'
        )


def main():
    parser = argparse.ArgumentParser(description='Monitor sync latency between image and pointcloud topics')
    parser.add_argument('--image-topic', default='/sensors/camera_0/color/image')
    parser.add_argument('--pointcloud-topic', default='/sensors/camera_0/points')
    parser.add_argument('--slop', type=float, default=0.1, help='Max allowed time difference (seconds)')
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = SyncLatencyMonitor(args.image_topic, args.pointcloud_topic, args.slop)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
