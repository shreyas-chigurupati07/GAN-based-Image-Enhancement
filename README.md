# Robotic Grasp Detection with GAN-Enhanced Depth Images

This repository contains the code and resources for a capstone project that improves robotic grasp detection through the enhancement of depth images using a Generative Adversarial Network (GAN). The project integrates advanced deep learning models within a ROS framework to refine depth images for precise robotic manipulation.

## Project Overview

Intel RealSense cameras provide valuable depth information for robotic applications but often suffer from noise that can degrade performance. To address this, our project employs a GAN to enhance the depth images, using the superior ZED2i camera images as ground truth. The enhanced images are subsequently fed into a GGCNN to accurately predict robotic grasp points.

## Dependencies

- ROS Noetic
- TensorFlow 2.x
- Keras 2.x
- OpenCV 4.x
- NumPy
- Python 3.x

## Installation

To set up the project environment, follow these steps:

1. **ROS Setup**:
   Ensure that ROS Noetic is installed and properly set up on your system. [ROS Installation Guide](http://wiki.ros.org/noetic/Installation)

2. **Python and Packages**:
   Install Python and the necessary packages using the following commands:
   
   ```bash
   sudo apt-get update
   sudo apt-get install python3-pip
   pip3 install tensorflow keras opencv-python numpy

3. **Usage**:
    roscore
    cd benchmarking_ws
    source devel/setup.bash
    rosrun gan_enhancer_node gan_enhancer_node.py

now run the Grasp algorithm by subscribing to the /enhanced_gan_depthimage topic


***Node Details***

gan_enhancer_node.py: A ROS node that subscribes to RealSense and ZED camera feeds to publish enhanced depth images.
***Subscriptions:***
/camera/depth/image_rect_raw: RealSense depth images.
/zed2i/zed_node/depth/depth_registered: ZED depth images.
***Publications:***
/enhanced_gan_depthimage: The topic where enhanced depth images are published.

**Model Information**

The trained GAN model is critical for the enhancement process and should be placed in a known directory. The default path is set to /home/merlab/Desktop/shreyas/best_generator.h5. Update the path in the gan_enhancer_node.py script if you store the model in a different location.

<p align="center">
   <img width="448" alt="Screenshot 2024-03-05 at 9 13 57 AM" src="https://github.com/shreyas-chigurupati07/GAN-based-Image-Enhancement/assets/84034817/7cf1c4c2-ad06-43a4-b61a-19be3d625cec">
</p>


**Dataset**

The dataset includes paired depth images from both the RealSense and ZED2i cameras. It is essential for training and validating the GAN model. Here is the link to the Dataset: https://wpi0-my.sharepoint.com/:f:/g/personal/schigurupati_wpi_edu/Ent9DaSGxR5AjIbc7_9k5U8BGRhI6nl76zaVLsNHlJ9T9w?e=plPdz4

**Results**

<img width="460" alt="Screenshot 2024-03-05 at 9 17 39 AM" src="https://github.com/shreyas-chigurupati07/GAN-based-Image-Enhancement/assets/84034817/e5e58a7d-0296-4f93-9e41-57d1fcd29c56">
<p align="center">
   <img width="470" alt="Screenshot 2024-03-05 at 9 17 58 AM" src="https://github.com/shreyas-chigurupati07/GAN-based-Image-Enhancement/assets/84034817/1da15689-be80-499f-9ab0-62a81acc460a">
</p>

**Scripts**
***main.py*** - This script has the complet GAN model used for Image enhancement
***zed_raw_capture.py*** - This script contains code for capturing depth images from the zed 2i camera module.
***realsense_capture.py*** - This script contains code for capturing raw depth images from the realsense camera module.
***grasp_prediction_nonenhanced.py*** - This script contains code for generating grasp rectangles on original non enhanced realsense depth image.
***grasp_prediction_enhanced.py*** - This script contains code for generating the grasp rectangles on the enhanced image output from the GAN model.
***gan_enhancer_node.py*** - This script contains code for ROS integration of the GAN model and GGCNN. 
