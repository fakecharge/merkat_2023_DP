#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, NavSatFix
import cv2
import numpy as np
from ultralytics import YOLO
from cv_bridge import CvBridge
import base64
from io import BytesIO
import requests
import time



class MyNode(Node):

    def __init__(self, test_image):
        super().__init__("first_node")
        topic_name= 'OAK_D_back/color/image'

        self.br = CvBridge()
        self.subscription = self.create_subscription(Image, topic_name, self.img_callback, 10)
        self.coor = []
        self.start = time.time()

        self.subscription_coor = self.create_subscription(
            NavSatFix,
            '/imu_sensor/imu/nav_sat_fix',
            self.listener_callback_coor,
            10)

        self.model = YOLO('/home/enchar/test_image/best.pt')
        
    
    def listener_callback_coor(self, data):
        self.coor = [ data.latitude, data.longitude]

    def img_callback(self, data):
        self.get_logger().info('Receiving video frame')
        current_frame = self.br.imgmsg_to_cv2(data)
        results = self.model.track(current_frame, device='cpu')

        result = results[0].plot()
        # result = current_frame
        # for next_box in results[0].boxes:
            # print(next_box.boxes)
            # cv2.rectangle(result, (int(next_box.xyxy[0][0]), int(next_box.xyxy[0][1])), (int(next_box.xyxy[0][2]), int(next_box.xyxy[0][3])), (0, 0, 255), 2, cv2.LINE_AA)

        image_data = cv2.imencode('.jpg', result)[1].tobytes()
        encoded_string = base64.b64encode(image_data).decode()
        headers = {
            'Content-Type': 'application/json',
        }
        data = {
            "image": encoded_string,
            "coordinate": self.coor
        }
        if time.time() - self.start > 1 and len(self.coor) > 0 and len(results[0].boxes) > 0:
            respones = requests.post(url="http://127.0.0.1:5000/test", headers=headers, json=data)
            self.get_logger().info(respones.text)
            self.start = time.time()

        
        cv2.imshow("camera", result)   
        cv2.waitKey(1)    
        
        
        

def main(args=None):
    rclpy.init(args=args)

    node = MyNode('/home/enchar/test_image/977224efd90d464487f7fe27922ae54b.jpeg')
    rclpy.spin(node)

    rclpy.shutdown()

if __name__ == '__main__':
    main()