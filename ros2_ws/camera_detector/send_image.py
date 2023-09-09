#!/usr/bin/env python3
import rclpy
import os
import glob
import time

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np

class SendImageNode(Node):
    def __init__(self, test_image):
        super().__init__("detector")
        # self.test = cv2.imread(test_image)
        self.images = glob.glob(test_image + "**.jpg", recursive=True)

        topic_name= 'video_frames'

        self.publisher_ = self.create_publisher(Image, topic_name , 10)
        self.br = CvBridge()

        self.timer = self.create_timer(0.1, self.timer_callback)


    def timer_callback(self):
        for next_file in self.images:
            self.test = cv2.imread(next_file)
            self.publisher_.publish(self.br.cv2_to_imgmsg(self.test))
            self.get_logger().info(f"publish image {next_file}")
            time.sleep(1)


def main(args=None):
    rclpy.init(args=args)
    simple_pub_sub = SendImageNode('/home/enchar/images/')
    rclpy.spin(simple_pub_sub)
    simple_pub_sub.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
