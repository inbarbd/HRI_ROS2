# import the opencv library
import cv2
import rclpy
from rclpy.node import Node
import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header , ColorRGBA



class publish_camera_fig(Node):

    def __init__(self,vid):
        super().__init__('publish_camera_fig')
        
        self.vid = vid
        self.bridge = CvBridge()
        self.publisher_ = self.create_publisher(Image, 'video_frames', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret == True:
                msg = self.bridge.cv2_to_imgmsg(frame)
                # print(msg.header.frame_id)
                # print(type(msg))
                # print(msg)
                msg.header.frame_id = str(self.i)
                msg.header.stamp = Node.get_clock(self).now().to_msg()
                self.publisher_.publish(msg)
                # cv2.imshow('frame', frame)
                self.get_logger().info('Publishing video frame')
        self.i += 1
        # print(self.i)

def main(args=None):
    vid = cv2.VideoCapture(0)
    rclpy.init(args=args)
    
    minimal_publisher = publish_camera_fig(vid)

    rclpy.spin(minimal_publisher)
    
    minimal_publisher.destroy_node()
    rclpy.shutdown()
    vid.release()
    

if __name__ == '__main__':
    main()