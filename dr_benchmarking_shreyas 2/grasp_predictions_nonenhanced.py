#!/usr/bin/env python3

import rospy
import numpy as np
import math
import cv2
from sensor_msgs.msg import Image
import time
from ggcnn.ggcnn_torch import predict, process_depth_image
from ggcnn.srv import GraspPrediction, GraspPredictionResponse

import cv_bridge
bridge = cv_bridge.CvBridge()


class GraspService:
    def __init__(self, sim_mode=False, crop=True):
        self.sim_mode = sim_mode
        self.crop = crop
        # Full image: [0, 0, 720, 1280]

        self.crop_size = [110, 185, 720, 1083]
        # self.crop_size = [110, 295, 720, 1181]
        # self.crop_size = [0,0,720,1280]
        self.min_frames_to_process = 50
        self.frames_recevied = 0
        self.safe_to_process = False

        if self.sim_mode:
            rospy.Subscriber("", Image, self.rgb_cb)
            rospy.Subscriber("", Image, self.depth_cb)
        else:
            rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_cb)
            # rospy.Subscriber("/camera/aligned_depth_to_color/depth_completed", Image, self.depth_cb)
            rospy.Subscriber(
                "/camera/aligned_depth_to_color/image_raw", Image, self.depth_cb)

        rospy.Service('debug', GraspPrediction, self.service_cb)

        self.rgb_cropped_pub = rospy.Publisher(
            "cropped_rgb", Image, queue_size=10)
        self.depth_cropped_pub = rospy.Publisher(
            "cropped_depth", Image, queue_size=10)
        print("Assignign Images to None")
        self.curr_depth_img = None
        self.curr_rgb_img = None

    def depth_cb(self, msg):

        img = bridge.imgmsg_to_cv2(msg)
        # print("Depth_Image Shape:{}".format(img.shape))
        if self.crop:
            # print("Im in Getting depth image - CROP")
            self.curr_depth_img = img[self.crop_size[0]:self.crop_size[2], self.crop_size[1]:self.crop_size[3]]
            self.depth_cropped_pub.publish(
                bridge.cv2_to_imgmsg(self.curr_depth_img))
        else:
            # print("Im in Getting depth image")
            self.curr_depth_img = img
        self.received = True
        self.frames_recevied += 1
        if self.frames_recevied >= self.min_frames_to_process:
            self.safe_to_process = True

    def rgb_cb(self, msg):
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # print("RGB Shape:{}".format(img.shape))
        if self.crop:
            self.curr_rgb_img = img[self.crop_size[0]:self.crop_size[2],
                                    self.crop_size[1]:self.crop_size[3], :]
            self.rgb_cropped_pub.publish(bridge.cv2_to_imgmsg(
                self.curr_rgb_img, encoding='bgr8'))
        else:
            self.curr_rgb_img = img


# Shreyas (custom call back) -start


    def service_cb(self, data):
        if not self.safe_to_process:
            rospy.logwarn(
                "Not enough frames received to safely process. Please wait...")
            return GraspPredictionResponse()

        depth = self.curr_depth_img
        rgb = self.curr_rgb_img

        # Process the depth image and predict the grasp using the GGCNN model.
        depth_crop, depth_nan_mask = process_depth_image(
            depth, depth.shape[0], 300, return_mask=True, crop_y_offset=0)
        points, angle, width_img, _ = predict(
            depth_crop, process_depth=False, depth_nan_mask=depth_nan_mask, filters=(2.0, 2.0, 2.0))

        # Find the grasp point in the image.
        x, y = np.unravel_index(np.argmax(points), points.shape)
        ang = angle[x][y]

        # Prepare the response.
        response = GraspPredictionResponse()
        g = response.best_grasp

        # Scale the grasp detection coordinates to the size of the depth image.
        g.pose.position.x = int(x * depth.shape[0] / 300)
        g.pose.position.y = int(
            y * depth.shape[0] / 300 + (depth.shape[1] - depth.shape[0]) / 2)
        g.pose.orientation.z = ang
        g.width = int(width_img[x][y] * depth.shape[0] / 300)

        rospy.loginfo(f"Grasp Detected - x: {x} y: {y}")
        rospy.loginfo(
            f"Accounting for crop - x: {g.pose.position.x} y: {g.pose.position.y}")

        # Normalize the depth image for display.
        depth_display = cv2.normalize(
            depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        depth_colored = cv2.cvtColor(depth_display, cv2.COLOR_GRAY2BGR)

        # Draw the rectangle on the depth image.
        depth_with_rect = self.draw_angled_rect(
            depth_colored, g.pose.position.y, g.pose.position.x, g.pose.orientation.z)

        # Display the depth image with the rectangle.
        cv2.imshow('Grasp Detection on Depth', depth_with_rect)
        # Use waitKey(1) to update the window properly within the ROS loop.
        cv2.waitKey(0)

        # Display additional information if necessary.
        points = cv2.cvtColor(points, cv2.COLOR_GRAY2RGB)
        angle = cv2.cvtColor(angle, cv2.COLOR_GRAY2RGB)
        points = cv2.circle(points, (y, x), 3, (0, 0, 255), -1)
        angle = cv2.circle(angle, (y, x), 3, (0, 0, 255), -1)
        cv2.imshow('Enhanced_Depth', depth_crop)
        cv2.imshow('Points', points)
        cv2.imshow('Angle', angle)

        return response
# Shreyas (custom call back) - end

    def draw_angled_rect(self, image, x, y, angle, width=100, height=50):
        """
        Draws bounding box for visualization
        """
        # Create a rotated rectangle
        b = math.cos(angle) * 0.5
        a = math.sin(angle) * 0.5

        display_image = image.copy()

        pt0 = (int(x - a * height - b * width),
               int(y + b * height - a * width))
        pt1 = (int(x + a * height - b * width),
               int(y - b * height - a * width))
        pt2 = (int(2 * x - pt0[0]), int(2 * y - pt0[1]))
        pt3 = (int(2 * x - pt1[0]), int(2 * y - pt1[1]))

        cv2.line(display_image, pt0, pt1, (0, 0, 255), 2)
        cv2.line(display_image, pt1, pt2, (0, 0, 255), 2)
        cv2.line(display_image, pt2, pt3, (0, 0, 255), 2)
        cv2.line(display_image, pt3, pt0, (0, 0, 255), 2)
        # cv2.circle(display_image, ((pt0[0] + pt2[0])//2, (pt0[1] + pt2[1])//2), 3, (0, 0, 0), -1)
        return display_image


if __name__ == '__main__':
    rospy.init_node('grasp_service')

    GraspService()
    rospy.spin()
