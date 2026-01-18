import cv2 as cv
import numpy as np
import torch

# Global variables to store depth and disparity maps
disparity_map = None
depth_map = None

# Function to handle mouse events
def on_mouse_click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:  # Check if the left mouse button was clicked
        if disparity_map is not None and depth_map is not None:
            disparity_value = disparity_map[y, x]
            depth_value = depth_map[y, x]
            print(f"Disparity at ({x}, {y}): {disparity_value}")
            print(f"Depth at ({x}, {y}): {depth_value:.2f} mm")

# Split the image frames
def frame_splitter(source_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, src_w, _ = source_image.shape
    hwd = int(src_w // 2)

    l_image = source_image[:, :hwd]
    r_image = source_image[:, hwd:]

    return l_image, r_image

# Load a pre-trained Depth Anything model (placeholder)
def apply_depth_anything(l_img):
    # Placeholder for actual Depth Anything model loading
    height, width = l_img.shape[:2]
    depth_anything_output = np.random.random((height, width)) * 255  # Placeholder for model output
    return depth_anything_output

# Function to create color bar for depth legend
def create_color_bar(colormap, height):
    color_bar = np.linspace(0, 255, 256).astype(np.uint8)
    color_bar = cv.applyColorMap(color_bar, colormap)
    color_bar = cv.resize(color_bar, (50, height))
    return color_bar

if __name__ == "__main__":

    # Open a connection to the camera (0 is usually the default camera)
    capture_device = cv.VideoCapture(0)

    # Check if the camera opened successfully
    if not capture_device.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Create a StereoSGBM object with updated parameters
    stereo = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,  # Must be divisible by 16
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    # Define the baseline distance (in mm)
    baseline = 45  # Baseline distance in mm

    # Define the focal length (in mm)
    focal_length = 400  # Example value, replace with your calculated focal length

    print("Starting, Press Q to exit..")

    while True:
        # Capture frame-by-frame
        ret, frame = capture_device.read()

        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        l_img, r_img = frame_splitter(frame)

        gl_img = cv.cvtColor(l_img, cv.COLOR_BGR2GRAY)
        gr_img = cv.cvtColor(r_img, cv.COLOR_BGR2GRAY)

        # Compute the disparity map with more detail
        disparity = stereo.compute(gl_img, gr_img).astype(np.float32)  # Convert to 32F (CV_32F)

        # --- Apply Median and Bilateral Filters to Disparity Map ---
        disparity = cv.medianBlur(disparity, 5)  # Apply median filter to reduce noise
        disparity_filtered = cv.bilateralFilter(disparity, d=9, sigmaColor=75, sigmaSpace=75)  # Bilateral filter

        # Normalize the disparity map for visualization
        disparity_filtered = cv.normalize(disparity_filtered, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        disparity_filtered = np.uint8(disparity_filtered)

        # Convert disparity to depth
        depth = (focal_length * baseline) / (disparity_filtered + 1e-6)

        # Apply Depth Anything model to refine depth
        depth_anything_output = apply_depth_anything(l_img)

        # Combine disparity and Depth Anything output (give more weight to Depth Anything)
        combined_depth = 0.3 * depth + 0.7 * depth_anything_output

        # Normalize combined depth for visualization
        depth_normalized = cv.normalize(combined_depth, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)

        # --- Masking based on combined depth map ---
        # Apply thresholding to create a binary mask
        mask_threshold_value = 100  # Adjust as needed
        _, mask = cv.threshold(depth_normalized, mask_threshold_value, 255, cv.THRESH_BINARY)

        # Apply the mask to the left image
        masked_image = cv.bitwise_and(l_img, l_img, mask=mask)

        # --- Further Noise Reduction on the Mask ---
        # Erode and dilate to remove small noise in the mask
        kernel = np.ones((3, 3), np.uint8)
        mask_eroded = cv.erode(mask, kernel, iterations=1)
        mask_cleaned = cv.dilate(mask_eroded, kernel, iterations=1)

        # Apply the cleaned mask to the left image
        masked_image_cleaned = cv.bitwise_and(l_img, l_img, mask=mask_cleaned)

        # Display the masked image and combined depth
        cv.imshow("Masked Image (Cleaned)", masked_image_cleaned)
        cv.imshow("Combined Depth (Refined)", depth_normalized)

        # Set mouse callback function to display disparity and depth values
        cv.setMouseCallback("Combined Depth (Refined)", on_mouse_click)

        # Break the loop on 'q' key press
        if cv.waitKey(1) == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    capture_device.release()
    cv.destroyAllWindows()
