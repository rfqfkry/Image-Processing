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
    _, src_w, _ = source_image.shape     # REFRENCE SHAPE (1440, 2880, 3)
    hwd = int(src_w // 2)                # LR (1440, 1440, 3)

    l_image = source_image[:, :hwd]
    r_image = source_image[:, hwd:]

    return l_image, r_image

if __name__ == "__main__":

    # Open a connection to the camera (0 is usually the default camera)
    capture_device = cv.VideoCapture(0)

    # Check if the camera opened successfully
    if not capture_device.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Create a StereoBM object
    stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)

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

        # Compute the disparity map
        disparity = stereo.compute(gl_img, gr_img)
        
        # Normalize the disparity map for visualization
        disparity = cv.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        disparity = np.uint8(disparity)

        # Apply Gaussian Blur to smooth the disparity map
        gaussian_blurred = cv.GaussianBlur(disparity, (5, 5), 0)

        # Apply Median Filter to smooth the disparity map
        median_filtered = cv.medianBlur(disparity, 5)

        # Convert disparity to depth
        # Adding a small value to avoid division by zero
        depth = (focal_length * baseline) / (disparity + 1e-6)

        # Update global variables
        disparity_map = disparity
        depth_map = depth

        rr_img = cv.resize(r_img, (512, 512))
        rl_img = cv.resize(l_img, (512, 512))

        rd_dip = cv.resize(disparity, (512, 512))
        rd_gauss = cv.resize(gaussian_blurred, (512, 512))
        rd_median = cv.resize(median_filtered, (512, 512))
        rd_dep = cv.resize(depth, (512, 512))

        cv.imshow("Left", rr_img)
        cv.imshow("Right", rl_img)
        cv.imshow("Disparity", rd_dip)
        cv.imshow("Gaussian Blurred Disparity", rd_gauss)
        cv.imshow("Median Filtered Disparity", rd_median)
        cv.imshow("Depth", rd_dep)

        # Set mouse callback function to display disparity and depth
        cv.setMouseCallback("Depth", on_mouse_click)

        # Break the loop on 'q' key press
        if cv.waitKey(1) == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    capture_device.release()
    cv.destroyAllWindows()
