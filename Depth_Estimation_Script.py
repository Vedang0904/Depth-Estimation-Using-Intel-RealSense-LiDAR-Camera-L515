import sys
import platform
import pyrealsense2 as rs
import numpy as np
import torch
import cv2
from torchvision import transforms

print("Python version:", sys.version)
print("Platform Python version:", platform.python_version())

try:
    import pyrealsense2 as rs
except ImportError:
    print("pyrealsense2 module not found. Please install it using 'pip install pyrealsense2'.")
    exit()

pipeline = rs.pipeline()
config = rs.config()

context = rs.context()
if len(context.devices) == 0:
    raise RuntimeError("No RealSense devices connected. Please check the connection.")

device = context.devices[0]
print(f"Using device: {device.get_info(rs.camera_info.name)}")

config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline_profile = pipeline.start(config)

try:
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
except Exception as e:
    print(f"Error loading MiDaS model: {e}")
    exit()

midas.eval()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

alpha = 0.2
depth_scale = 1.0  # You might need to calibrate this value

def apply_ema_filter(current_depth, previous_depth):
    return alpha * current_depth + (1 - alpha) * previous_depth

def depth_to_distance(depth_value, depth_scale):
    if depth_value == 0:  # Handle potential zero depth values to avoid division by zero
        return 0.0
    return 1.0 / (depth_value * depth_scale)

def calculate_rmse(predictions, targets):
    """Calculates Root Mean Squared Error."""
    return np.sqrt(np.mean((predictions - targets)**2))

def calculate_mae(predictions, targets):
    """Calculates Mean Absolute Error."""
    return np.mean(np.abs(predictions - targets))

previous_depth_midas = 0.0
previous_depth_realsense = 0.0

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        

        input_batch = transform(color_image).unsqueeze(0)
        with torch.no_grad():
            prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=color_image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth_map = prediction.cpu().numpy()

        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
        
        h_midas, w_midas = depth_map.shape
        mid_x_midas = w_midas // 2
        mid_y_midas = h_midas // 2
        depth_value_midas = depth_map[mid_y_midas, mid_x_midas]

        h_realsense, w_realsense = depth_image.shape
        mid_x_realsense = w_realsense // 2
        mid_y_realsense = h_realsense // 2
        depth_value_realsense = depth_image[mid_y_realsense, mid_x_realsense]
        
        depth_map_resized = cv2.resize(depth_map, (depth_image.shape[1], depth_image.shape[0]))
        rmse = calculate_rmse(depth_map_resized, depth_image)
        mae = calculate_mae(depth_map_resized, depth_image)
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        distance_midas = depth_to_distance(depth_value_midas, depth_scale)
        filtered_distance_midas = apply_ema_filter(distance_midas, previous_depth_midas)
        previous_depth_midas = filtered_distance_midas

        distance_realsense = depth_to_distance(depth_value_realsense, depth_scale)


        filtered_distance_realsense = apply_ema_filter(distance_realsense, previous_depth_realsense)
        previous_depth_realsense = filtered_distance_realsense

        cv2.putText(depth_map_colored, f"MiDaS Distance: {filtered_distance_midas:.2f} units", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_image_colored = cv2.applyColorMap(depth_image_normalized, cv2.COLORMAP_JET)
        cv2.putText(depth_image_colored, f"RealSense Distance: {filtered_distance_midas:.2f} units", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        depth_image_colored_resized = cv2.resize(depth_image_colored, (depth_map_colored.shape[1], depth_map_colored.shape[0]))
        combined_view = np.hstack((depth_image_colored_resized, depth_map_colored))
        cv2.imshow("RealSense Depth vs MiDaS Depth", combined_view)

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()