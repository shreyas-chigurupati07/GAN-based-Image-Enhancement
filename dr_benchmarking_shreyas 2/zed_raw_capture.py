#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import pyzed.sl as sl
import os

def main():
    rospy.init_node('zed_camera_publisher_node', anonymous=True)
    rospy.loginfo("ZED Camera Publisher Node Started")
    
    image_pub = rospy.Publisher('/zed_camera/image', Image, queue_size=1)
    depth_pub = rospy.Publisher('/zed_camera/depth', Image, queue_size=1)
    
    bridge = CvBridge()
    zed = sl.Camera()
    visualize_one_image = True
    
    input_type = sl.InputType()
    init_params = sl.InitParameters(input_t=input_type)
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL 
    init_params.coordinate_units = sl.UNIT.METER  
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.depth_minimum_distance = 0.2 
    runtime_parameters =sl.RuntimeParameters()
    runtime_parameters.enable_fill_mode = True
    # init_params.coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD


    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        zed.close()
        exit(1)

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_configuration.resolution
    width = image_size.width
    height = image_size.height

    image_zed = sl.Mat()
    depth_image_zed = sl.Mat(width, height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)



    # Define the cropping area (example: x, y, width, height)
    crop_area = (475, 100, 450, 400)  # Adjust these values as needed

    # Directory to save images
    save_directory = "/home/merlab/Desktop/shreyas/depth_images"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Counter for the number of images captured
    image_counter = 0

    while not rospy.is_shutdown() and image_counter < 50:
        err = zed.grab(runtime_parameters)
        if err == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU)
            zed.retrieve_measure(depth_image_zed, sl.MEASURE.DEPTH, sl.MEM.CPU)
           
            image_ocv = image_zed.get_data()
            depth_image_ocv = depth_image_zed.get_data()

            # Normalize one image for visualization purposes
            if visualize_one_image:
                normalized_depth_image = cv2.normalize(depth_image_ocv, None, 0, 255, cv2.NORM_MINMAX)
                normalized_depth_image = cv2.convertScaleAbs(normalized_depth_image)
                cv2.imshow("Normalized Depth Image", normalized_depth_image)
                cv2.waitKey(0)
                visualize_one_image = False  # Turn off visualization after the first image

            # Crop the depth image
            x, y, w, h = crop_area
            cropped_depth_image = depth_image_ocv[y:y+h, x:x+w]

            # Save the cropped depth image
            depth_image_filename = os.path.join(save_directory, f"cropped_depth_image_{image_counter:03d}.png")
            cv2.imwrite(depth_image_filename, cropped_depth_image)
            image_counter += 1

            # Publish the images to ROS topics
            try:
                image_msg = bridge.cv2_to_imgmsg(image_ocv, encoding="bgra8")
                depth_msg = bridge.cv2_to_imgmsg(cropped_depth_image, encoding="32FC1")
                image_pub.publish(image_msg)
                depth_pub.publish(depth_msg)
            except CvBridgeError as e:
                print(e)

        else:
            rospy.loginfo(f"Failed to grab frame: {str(err)}")
        rospy.sleep(0.1)  # To prevent the script from overloading the CPU

    zed.close()

if __name__ == "__main__":
    main()