import os
import cv2
import numpy as np
import math
import scipy.ndimage as ndimage
import torch
from matplotlib import pyplot as plt
import importlib.util
import sys

spec = importlib.util.spec_from_file_location(
    "ggcnn", "/content/drive/MyDrive/benchmarking_project/ggcnn.py")
ggcnn = importlib.util.module_from_spec(spec)
sys.modules["ggcnn"] = ggcnn
spec.loader.exec_module(ggcnn)

model = ggcnn.GGCNN()

# Load the model state dictionary
MODEL_STATE_DICT_FILE = '/content/drive/MyDrive/benchmarking_project/ggcnn_epoch_23_cornell_statedict.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the model
model.load_state_dict(torch.load(MODEL_STATE_DICT_FILE, map_location=device))
model.to(device)
model.eval()


def process_depth_image(depth, crop_size, out_size=300, return_mask=False, crop_y_offset=0):
    imh, imw = depth.shape

    depth_crop = depth[(imh - crop_size) // 2 - crop_y_offset:(imh - crop_size) // 2 + crop_size - crop_y_offset,
                       (imw - crop_size) // 2:(imw - crop_size) // 2 + crop_size]

    depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)

    depth_crop[depth_nan_mask == 1] = 0

    depth_scale = np.abs(depth_crop).max()

    depth_crop = depth_crop.astype(np.float32) / depth_scale

    depth_crop = cv2.inpaint(depth_crop, depth_nan_mask, 1, cv2.INPAINT_NS)

    depth_crop = depth_crop[1:-1, 1:-1]
    depth_crop = depth_crop * depth_scale

    depth_crop = cv2.resize(
        depth_crop, (out_size, out_size), interpolation=cv2.INTER_AREA)

    if return_mask:

        depth_nan_mask = depth_nan_mask[1:-1, 1:-1]
        depth_nan_mask = cv2.resize(
            depth_nan_mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        return depth_crop, depth_nan_mask
    else:
        return depth_crop


def predict(depth, process_depth=True, out_size=300, filters=(2.0, 1.0, 1.0)):

    depth_resized = cv2.resize(
        depth, (out_size, out_size), interpolation=cv2.INTER_AREA)

    depthT = torch.from_numpy(depth_resized.reshape(
        1, 1, out_size, out_size).astype(np.float32)).to(device)

    with torch.no_grad():
        pred_out = model(depthT)

    # Extract the outputs for grasp point, angle, and width
    points_out = pred_out[0].cpu().numpy().squeeze()
    cos_out = pred_out[1].cpu().numpy().squeeze()
    sin_out = pred_out[2].cpu().numpy().squeeze()
    ang_out = np.arctan2(sin_out, cos_out) / 2.0
    width_out = pred_out[3].cpu().numpy().squeeze() * 150.0

    if filters[0]:
        points_out = ndimage.gaussian_filter(
            points_out, filters[0])
    if filters[1]:
        ang_out = ndimage.gaussian_filter(ang_out, filters[1])
    if filters[2]:
        width_out = ndimage.gaussian_filter(width_out, filters[2])

    points_out = np.clip(points_out, 0.0, 1.0-1e-3)

    return points_out, ang_out, width_out, depth_resized


def draw_grasp_rectangle(image, grasp_point, angle, width, scale=1.0):

    display_width = width * scale
    display_length = display_width * 2

    angle_degrees = angle * 180.0 / math.pi

    grasp_point = (float(grasp_point[0]), float(grasp_point[1]))

    box = ((grasp_point[0], grasp_point[1]),
           (display_length, display_width), angle_degrees)

    rect = cv2.boxPoints(box)
    rect = np.int0(rect)

    cv2.drawContours(image, [rect], 0, (0, 0, 255), 2)
    return image


DEPTH_IMAGE_PATH = '/content/drive/MyDrive/benchmarking_project/enhanced_images_output/013.png'


original_depth_img = cv2.imread(DEPTH_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

if original_depth_img is None:
    raise ValueError(f"Image at {DEPTH_IMAGE_PATH} could not be loaded.")


if len(original_depth_img.shape) > 2 and original_depth_img.shape[2] > 1:
    original_depth_img = cv2.cvtColor(original_depth_img, cv2.COLOR_BGR2GRAY)


points_out, ang_out, width_out, _ = predict(
    original_depth_img, process_depth=False)


grasp_point_idx = np.unravel_index(np.argmax(points_out), points_out.shape)
grasp_point = (grasp_point_idx[1], grasp_point_idx[0])
angle = ang_out[grasp_point_idx]
width = width_out[grasp_point_idx]


if len(original_depth_img.shape) == 2 or original_depth_img.shape[2] == 1:
    original_color_img = cv2.cvtColor(original_depth_img, cv2.COLOR_GRAY2BGR)
grasp_img = draw_grasp_rectangle(
    original_color_img, grasp_point, angle, width, scale=0.2)

# Display the results
plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.title('Original Depth Image')
plt.imshow(original_depth_img, cmap='gray')
plt.subplot(222)
plt.title('Processed Depth Image')
plt.imshow(original_depth_img, cmap='gray')
plt.subplot(223)
plt.title('Grasp Prediction Points')
plt.imshow(points_out, cmap='jet')
plt.subplot(224)
plt.title('Grasp Rectangle')
plt.imshow(grasp_img)
plt.show()
