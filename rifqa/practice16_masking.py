import cv2 as cv
import numpy as np
import torch

device     = torch.device('cpu')

MIDAS_MODEL = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
MIDAS_MODEL.eval()

MIDAS_TRANFORM = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
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
def compute_midas(source_image : np.ndarray) -> np.ndarray:
    source_image = cv.cvtColor(source_image, cv.COLOR_BGR2RGB)
    with torch.no_grad():
        input_batch : torch.Tensor = MIDAS_TRANFORM(source_image).to(device)
    
        prediction : torch.Tensor  = MIDAS_MODEL(input_batch)

        prediction : torch.Tensor  = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size          = source_image.shape[:2],
            mode          = "bicubic",
            align_corners = False,
        )

        prediction : torch.Tensor  = torch.squeeze(prediction)

    output : np.ndarray = prediction.cpu().numpy()
    return output

def compute_region_grayscale(source_image : np.ndarray) -> np.ndarray:

    midas_output = compute_midas(source_image) # (1440, 1400)

    midas_output : np.ndarray = (midas_output - np.min(midas_output)) / np.ptp(midas_output)
    midas_output : np.ndarray = midas_output * 100
    midas_output : np.ndarray = np.round(midas_output)
    midas_output : np.ndarray = midas_output.astype(np.uint8)

    return midas_output

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

        # left_midas  = compute_region_grayscale(left_img)
        # right_midas = compute_region_grayscale(right_img)


        # Compute the disparity map with more detail
        disparity = stereo.compute(gl_img, gr_img)
        
        # Normalize the disparity map for visualization
        disparity = cv.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        disparity = np.uint8(disparity)
        print(np.min(disparity),np.max(disparity))
        
        # Convert disparity to depth
        depth = (focal_length * baseline) / (disparity + 1e-6)
        depth = cv.normalize(depth, depth , alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        depth  = np.uint8(depth)
        print(np.min(depth),np.max(depth))

        # Apply thresholding to the disparity map ______COBA GANTI ANGKA THRESHOLDNYA________
        threshold_value = 128  # Adjust this value as needed, 
        _, dep_thresholded = cv.threshold(depth, threshold_value, 255, cv.THRESH_BINARY) #cari dikelas threshold kenapa dia ada 4 argumen
        dep_thresholded_resized = cv.resize(dep_thresholded, (512, 512))
        '''Threshold adalah nilai yang ditentukan untuk memberi batasan pada suatu pixel (segmentasi lah kurleb)'''

        # Apply bilateral filter to smooth the thresholded disparity map
        dep_filtered = cv.bilateralFilter(dep_thresholded, d=9, sigmaColor=75, sigmaSpace=75) #cari dikelas bilateral 

        # Apply a colormap to the filtered disparity map
        colormap_dep = cv.applyColorMap(dep_filtered, cv.COLORMAP_JET)

        # Normalize depth for visualization (closer objects brighter, farther objects darker)
        #depth itu ga terbatas, jadi tujuann disini adalah untuk memberi batasan depth berdasarkan
        # depth_normalized = cv.normalize(depth, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        # depth_normalized = np.uint8(depth_normalized)

        # Update global variables
        disparity_map = disparity
        depth_map = depth

        rr_img = cv.resize(r_img, (512, 512))
        rl_img = cv.resize(l_img, (512, 512))

        rd_dip = cv.resize(disparity, (512, 512))
        rd_colormap_dep = cv.resize(colormap_dep, (512, 512))
        # rd_dep = cv.resize(depth_normalized, (512, 512))

        # Create color bar with the same height as rd_colormap_dip
        color_bar = create_color_bar(cv.COLORMAP_JET, rd_colormap_dep.shape[0])#array ke shape 0, shape ini isinya ada 3 array

        # Combine color bar with the colored disparity image
        combined_display = cv.hconcat([rd_colormap_dep, color_bar])

        # Add text to the color bar for legend
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(combined_display, 'Far', (rd_colormap_dep.shape[1] + 5, 20), font, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(combined_display, 'Near', (rd_colormap_dep.shape[1] + 5, combined_display.shape[0] - 10), font, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        # --- Masking Blue Color in Disparity Map ---

        # Convert the disparity image to HSV color space
        hsv = cv.cvtColor(rd_colormap_dep, cv.COLOR_BGR2HSV)

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
       
        left_midas  = compute_region_grayscale(masked_left_resized)

       
        left_midas = cv.normalize(left_midas, left_midas , alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        left_midas  = np.uint8(left_midas)
        print(np.min(left_midas),np.max(left_midas))

        # Apply thresholding to the disparity map ______COBA GANTI ANGKA THRESHOLDNYA________
        threshold_value = 128  # Adjust this value as needed, 
        _, midas_thresholded = cv.threshold(left_midas, threshold_value, 255, cv.THRESH_BINARY) #cari dikelas threshold kenapa dia ada 4 argumen
        midas_thresholded_resized = cv.resize(midas_thresholded, (512, 512))
        '''Threshold adalah nilai yang ditentukan untuk memberi batasan pada suatu pixel (segmentasi lah kurleb)'''

        # Apply bilateral filter to smooth the thresholded disparity map
        midas_filtered = cv.bilateralFilter(midas_thresholded, d=9, sigmaColor=75, sigmaSpace=75) #cari dikelas bilateral 

        # Apply a colormap to the filtered disparity map
        colormap_midas = cv.applyColorMap(midas_filtered, cv.COLORMAP_JET)

        hsv_midas = cv.cvtColor(colormap_midas, cv.COLOR_BGR2HSV)
        
        masked_midas = cv.inRange(hsv_midas, lower_blue, upper_blue)

        # Invert the mask
        inverse_maskmidas = cv.bitwise_not(masked_midas)

        # Apply the mask to remove the blue regions
        masked_resultr = cv.bitwise_and(rr_img, rr_img, mask=inverse_maskmidas)
        masked_resultl = cv.bitwise_and(rl_img, rl_img, mask=inverse_maskmidas)
        masked_left_resized = cv.resize(masked_resultl, (512, 512))

       #membandingkan dimensi
        if len(rl_img.shape) == 2:  # If image is grayscale
            rl_img = cv.cvtColor(rl_img, cv.COLOR_GRAY2BGR)
        if len(dep_thresholded_resized.shape) == 2:#jika dia dimensinya 2, maka ditambahkan colorbgr jadilah 3d 
            dep_thresholded_resized = cv.cvtColor(dep_thresholded_resized, cv.COLOR_GRAY2BGR)
        if len(masked_left_resized.shape) == 2:
            masked_left_resized = cv.cvtColor(masked_left_resized, cv.COLOR_GRAY2BGR)       

        #combined the original file left, disparity, masking left
        combined_display_left = cv.hconcat([rl_img, dep_thresholded_resized, masked_left_resized])

       
        # --- Display Results ---

        #cv.imshow("Left", rr_img)
        cv.imshow("Right", rl_img)
        # cv.imshow("Disparity", masked_midas)
        # cv.imshow("Colored Disparity with Legend", combined_display)  # Display the colored disparity map with legend
        #cv.imshow("Depth", rd_dep)
        # cv.imshow("Combined Left, Disparity, Masked Left", combined_display_left)
    


        # Display the masked disparity map
        #cv.imshow('Masked right', masked_resultr)
        #cv.imshow('masked left',masked_resultl)
        #cv.imshow('Original Frame', frame)          

        # Set mouse callback function to display disparity and depth
        # cv.setMouseCallback("Combined Left, Disparity, Masked Left", on_mouse_click)

        # Break the loop on 'q' key press
        if cv.waitKey(1) == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    capture_device.release()
    cv.destroyAllWindows()
