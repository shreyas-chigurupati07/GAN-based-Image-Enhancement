import pyrealsense2 as rs
import numpy as np
import cv2

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable the depth stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the pipeline
pipeline.start(config)

# Define the number of images to capture
num_images = 60

# Define the cropping area (example: x, y, width, height)
crop_area = (200, 10, 300, 400)  # Adjust the values to your specific FOV

try:
    for image_count in range(num_images):
        # Wait for a coherent pair of frames: depth
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())

        # Crop the depth image
        x, y, w, h = crop_area
        cropped_depth_image = depth_image[y:y+h, x:x+w]

        # Save the cropped depth image to a file
        filename = f'cropped_depth_image_pose0_{image_count+1}.png'
        cv2.imwrite(filename, cropped_depth_image)

        print(f"Saved {filename}")

except Exception as e:
    print(e)
finally:
    # Stop the pipeline
    pipeline.stop()
