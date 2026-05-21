#!/usr/bin/env python3
import argparse

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image


class SyncLatencyMonitorImages(Node):
    def __init__(self, image_topic_a, image_topic_b, slop):
        super().__init__('sync_latency_monitor_images')
        self.sub_a = Subscriber(self, Image, image_topic_a)
        self.sub_b = Subscriber(self, Image, image_topic_b)
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_a, self.sub_b], queue_size=10, slop=slop
        )
        self.sync.registerCallback(self.callback)
        self.get_logger().info(
            f'Listening on "{image_topic_a}" and "{image_topic_b}" '
            f'with slop={slop}s'
        )

    def callback(self, msg_a, msg_b):
        now = self.get_clock().now()
        t_a = Time.from_msg(msg_a.header.stamp)
        t_b = Time.from_msg(msg_b.header.stamp)

        diff_ns = abs((t_a - t_b).nanoseconds)
        oldest = min(t_a, t_b)
        latency_ns = (now - oldest).nanoseconds

        self.get_logger().info(
            f'Color-Depth diff: {diff_ns / 1e6:.2f} ms | '
            f'Oldest->now latency: {latency_ns / 1e6:.2f} ms'
        )


def main():
    parser = argparse.ArgumentParser(description='Monitor sync latency between two image topics')
    parser.add_argument('--image-topic-a', default='/sensors/camera_0/color/image')
    parser.add_argument('--image-topic-b', default='/sensors/camera_0/depth/image')
    parser.add_argument('--slop', type=float, default=0.1, help='Max allowed time difference (seconds)')
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = SyncLatencyMonitorImages(args.image_topic_a, args.image_topic_b, args.slop)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
