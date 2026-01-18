import cv2 as cv
import numpy as np

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
    _, src_w, _ = source_image.shape     # REFERENCE SHAPE (1440, 2880, 3)
    hwd = int(src_w // 2)                # LR (1440, 1440, 3)

    l_image = source_image[:, :hwd]
    r_image = source_image[:, hwd:]

    return l_image, r_image

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

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break
        
        l_img, r_img = frame_splitter(frame)

        gl_img = cv.cvtColor(l_img, cv.COLOR_BGR2GRAY)
        gr_img = cv.cvtColor(r_img, cv.COLOR_BGR2GRAY)

        # Compute the disparity map with more detail
        disparity = stereo.compute(gl_img, gr_img)
        
        # Normalize the disparity map for visualization
        disparity = cv.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        disparity = np.uint8(disparity)

        # Apply thresholding to the disparity map
        threshold_value = 100  # Adjust this value as needed
        _, disparity_thresholded = cv.threshold(disparity, threshold_value, 255, cv.THRESH_BINARY)
        disparity_thresholded_resized = cv.resize(disparity_thresholded, (512, 512))

        # Apply bilateral filter to smooth the thresholded disparity map
        disparity_filtered = cv.bilateralFilter(disparity_thresholded, d=9, sigmaColor=75, sigmaSpace=75)

        # Apply a colormap to the filtered disparity map
        colormap_disparity = cv.applyColorMap(disparity_filtered, cv.COLORMAP_JET)

        # Convert disparity to depth
        depth = (focal_length * baseline) / (disparity + 1e-6)

        # Normalize depth for visualization (closer objects brighter, farther objects darker)
        depth_normalized = cv.normalize(depth, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)


        # Apply bilateral filter to smooth the disparity map
        disparity_filtered = cv.bilateralFilter(disparity, d=9, sigmaColor=75, sigmaSpace=75)

        # Apply a colormap to the filtered disparity map
        colormap_disparity = cv.applyColorMap(disparity_filtered, cv.COLORMAP_JET)

        # Convert disparity to depth
        depth = (focal_length * baseline) / (disparity + 1e-6)

        # Normalize depth for visualization (closer objects brighter, farther objects darker)
        depth_normalized = cv.normalize(depth, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)

        # Update global variables
        disparity_map = disparity
        depth_map = depth

        rr_img = cv.resize(r_img, (512, 512))
        rl_img = cv.resize(l_img, (512, 512))

        rd_dip = cv.resize(disparity, (512, 512))
        rd_colormap_dip = cv.resize(colormap_disparity, (512, 512))
        rd_dep = cv.resize(depth_normalized, (512, 512))

        # Create color bar with the same height as rd_colormap_dip
        color_bar = create_color_bar(cv.COLORMAP_JET, rd_colormap_dip.shape[0])

        # Combine color bar with the colored disparity image
        combined_display = cv.hconcat([rd_colormap_dip, color_bar])

        # Add text to the color bar for legend
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(combined_display, 'Far', (rd_colormap_dip.shape[1] + 5, 20), font, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(combined_display, 'Near', (rd_colormap_dip.shape[1] + 5, combined_display.shape[0] - 10), font, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        # --- Masking Blue Color in Disparity Map ---

        # Convert the disparity image to HSV color space
        hsv = cv.cvtColor(rd_colormap_dip, cv.COLOR_BGR2HSV)

        # Define range of blue color in HSV
        lower_blue = np.array([60, 40, 40])
        upper_blue = np.array([130, 255, 255])

        # Create a mask for the blue color
        mask = cv.inRange(hsv, lower_blue, upper_blue)

        # Invert the mask
        inverse_mask = cv.bitwise_not(mask)

        # Apply the mask to remove the blue regions
        masked_resultr = cv.bitwise_and(rr_img, rr_img, mask=inverse_mask)
        masked_resultl = cv.bitwise_and(rl_img, rl_img, mask=inverse_mask)
        masked_left_resized = cv.resize(masked_resultl, (512, 512))

        #combined the original file left, disparity, masking left
        combined_display_left = cv.hconcat([rl_img, disparity_thresholded_resized, masked_left_resized])

       
        # --- Display Results ---

        cv.imshow("Left", rr_img)
        cv.imshow("Right", rl_img)
        cv.imshow("Disparity", rd_dip)
        cv.imshow("Colored Disparity with Legend", combined_display)  # Display the colored disparity map with legend
        cv.imshow("Depth", rd_dep)
        cv.imshow("Combined Left, Disparity, Masked Left", combined_display_left)


        # Display the masked disparity map
        cv.imshow('Masked right', masked_resultr)
        cv.imshow('masked left',masked_resultl)
        cv.imshow('Original Frame', frame)          

        # Set mouse callback function to display disparity and depth
        cv.setMouseCallback("Depth", on_mouse_click)

        # Break the loop on 'q' key press
        if cv.waitKey(1) == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    capture_device.release()
    cv.destroyAllWindows()
