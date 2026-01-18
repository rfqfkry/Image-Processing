import cv2 as cv
import numpy as np
import torch
import matplotlib.pyplot as plt

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

# Function to apply MiDaS for depth estimation
def apply_midas(frame, midas, transform):
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Convert image to RGB for MiDaS
    imgbatch = transform(img).to('cpu')         # Apply transformations

    # Make a prediction using MiDaS
    with torch.no_grad(): 
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2], 
            mode='bicubic', 
            align_corners=False
        ).squeeze()

        depth_output = prediction.cpu().numpy()
        return depth_output

if __name__ == "__main__":

    # Load MiDaS model
    midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
    midas.to('cpu')
    midas.eval()

    # Input transformation pipeline for MiDaS
    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    transform = transforms.small_transform

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

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break
        
        # Split the stereo image into left and right images
        l_img = frame[:, :frame.shape[1]//2]
        r_img = frame[:, frame.shape[1]//2:]

        gl_img = cv.cvtColor(l_img, cv.COLOR_BGR2GRAY)
        gr_img = cv.cvtColor(r_img, cv.COLOR_BGR2GRAY)

        # Compute the disparity map with more detail
        disparity = stereo.compute(gl_img, gr_img)
        
        # Normalize the disparity map for visualization
        disparity = cv.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        disparity = np.uint8(disparity)

        # Convert disparity to depth
        depth = (focal_length * baseline) / (disparity + 1e-6)

        # Normalize depth for visualization (closer objects brighter, farther objects darker)
        depth_normalized = cv.normalize(depth, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)

        # Use left image as input for MiDaS
        depth_midas = apply_midas(l_img, midas, transform)

        # Convert MiDaS depth to uint8 for display
        depth_midas_normalized = cv.normalize(depth_midas, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        depth_midas_normalized = np.uint8(depth_midas_normalized)

        # --- Combine Depth Maps (Stereo and MiDaS) ---
        # Normalize the disparity and MiDaS depth maps to [0, 1] range
        disparity_normalized = cv.normalize(disparity, None, 0, 1, cv.NORM_MINMAX)
        depth_midas_normalized_01 = cv.normalize(depth_midas_normalized, None, 0, 1, cv.NORM_MINMAX)

        # Combine using a weighted average or minimum
        combined_depth = (0.5 * disparity_normalized + 0.5 * depth_midas_normalized_01)

        # Threshold the combined depth for masking
        threshold_value = 0.5  # Adjust this value based on your requirements
        _, mask = cv.threshold(combined_depth, threshold_value, 1, cv.THRESH_BINARY)

        # Convert the mask to 8-bit for visualization and apply to the original left image
        mask = (mask * 255).astype(np.uint8)
        masked_image = cv.bitwise_and(l_img, l_img, mask=mask)

        # --- Display Results ---
        cv.imshow("Left Image", l_img)
        cv.imshow("Disparity Map", disparity)
        cv.imshow("Depth (Stereo from Disparity)", depth_normalized)
        cv.imshow("Depth (MiDaS on Left Image)", depth_midas_normalized)  # Show depth from MiDaS using left image
        cv.imshow("Combined Depth Mask", masked_image)  # Show the mask from combined depth

        # Set mouse callback function to display disparity and depth
        cv.setMouseCallback("Combined Depth Mask", on_mouse_click)

        # Break the loop on 'q' key press
        if cv.waitKey(1) == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    capture_device.release()
    cv.destroyAllWindows()
