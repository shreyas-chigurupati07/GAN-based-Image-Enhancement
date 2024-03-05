#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from tensorflow.keras.models import load_model
import numpy as np


class GANEnhancerNode:
    def __init__(self):
        rospy.init_node('gan_enhancer_node', anonymous=True)

        self.generator_model = load_model(
            '/home/merlab/Desktop/shreyas/best_generator.h5')

        self.bridge = CvBridge()

        # Subscribers for RealSense and ZED images
        self.depth_sub = rospy.Subscriber(
            '/camera/depth/image_rect_raw', Image, self.depth_callback)
        self.zed_sub = rospy.Subscriber(
            '/zed2i/zed_node/depth/depth_registered', Image, self.zed_callback)

        self.enhanced_pub = rospy.Publisher(
            '/enhanced_gan_depthimage', Image, queue_size=10)

        # Store the latest ZED image
        self.latest_zed_image = None

    def preprocess_image(self, cv_image):
        # Resize to match model's expected input size
        image_resized = cv2.resize(cv_image, (640, 480))
        image_normalized = (image_resized / 255.0) * 2 - 1
        image_normalized = image_normalized.astype(np.float32)
        image_expanded = np.expand_dims(
            image_normalized, axis=0)  # Add batch dimension
        image_expanded = np.expand_dims(
            image_expanded, axis=-1)  # Add channel dimension
        return image_expanded

    def zed_callback(self, data):
        try:
            # Convert the ZED depth image to CV2 format
            self.latest_zed_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            rospy.logerr(f"ZED CV Bridge error: {e}")

    def depth_callback(self, data):
        try:
            if self.latest_zed_image is None:
                rospy.logwarn("No ZED image received yet")
                return

            # Convert RealSense Image to CV2 format
            cv_image = self.bridge.imgmsg_to_cv2(data, "32FC1")

            # Preprocess RealSense and ZED images
            preprocessed_depth_image = self.preprocess_image(cv_image)
            preprocessed_zed_image = self.preprocess_image(
                self.latest_zed_image)

            # Use GAN model to enhance image
            enhanced_image = self.generator_model.predict(
                [preprocessed_depth_image, preprocessed_zed_image])

            # Post-process the enhanced image
            enhanced_image = (enhanced_image + 1) / 2 * 255
            enhanced_image = enhanced_image.astype(np.uint8)

            # Convert back to ROS Image message and publish
            enhanced_msg = self.bridge.cv2_to_imgmsg(
                enhanced_image[0, :, :, 0], "mono8")
            self.enhanced_pub.publish(enhanced_msg)

        except CvBridgeError as e:
            rospy.logerr(f"RealSense CV Bridge error: {e}")


if __name__ == '__main__':
    node = GANEnhancerNode()
    rospy.spin()
