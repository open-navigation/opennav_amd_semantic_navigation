#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class Viewer(Node):
    def __init__(self):
        super().__init__('image_viewer')
        self.bridge = CvBridge()
        self.fullscreen = False
        self.frames_receiving = False
        self.sub = self.create_subscription(Image, '~/image', self.cb, 1)

    def cb(self, msg):
        if self.frames_receiving == False:
            cv2.namedWindow('SAM3 Demo', cv2.WINDOW_NORMAL)  # WINDOW_NORMAL = resizable, image scales to window
            self.frames_receiving = True
            
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv2.imshow('SAM3 Demo', img)
        key = cv2.waitKey(1)
        if key == ord('f'):
            self.fullscreen = not self.fullscreen
            cv2.setWindowProperty('SAM3 Demo', cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)

def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(Viewer())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
