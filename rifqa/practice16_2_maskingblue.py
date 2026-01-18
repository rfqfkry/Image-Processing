import cv2 as cv
import numpy as np
import torch
import time 
from threading import Thread # library for implementing multi-threaded processing 

device     = torch.device('cpu')

MIDAS_MODEL = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
MIDAS_MODEL.eval()

MIDAS_TRANFORM = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
# Global variables to store depth and disparity maps
disparity_map = None
depth_map = None

# defining a helper class for implementing multi-threaded processing 
class WebcamStream :
    def __init__(self, stream_id=0):
        self.stream_id = stream_id   # default is 0 for primary camera 
        
        # opening video capture stream 
        self.vcap      = cv.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False :
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5))
        print("FPS of webcam hardware/input stream: {}".format(fps_input_stream))
            
        # reading a single frame from vcap stream for initializing 
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)

        # self.stopped is set to False when frames are being read from self.vcap stream 
        self.stopped = True 

        # reference to the thread for reading next available frame from input stream 
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True # daemon threads keep running in the background while the program is executing 
        
    # method for starting the thread for grabbing next available frame in input stream 
    def start(self):
        self.stopped = False
        self.t.start() 

    # method for reading next frame 
    def update(self):
        while True :
            if self.stopped is True :
                break
            self.grabbed , self.frame = self.vcap.read()
            if self.grabbed is False :
                print('[Exiting] No more frames to read')
                self.stopped = True
                break 
        self.vcap.release()

    # method for returning latest read frame 
    def read(self):
        return self.frame

    # method called to stop reading frames 
    def stop(self):
        self.stopped = True 

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
    capture_device = WebcamStream(stream_id=0) #  stream_id = 0 is for primary camera 
    capture_device.start()



    # Create a StereoSGBM object with updated parameters
    stereo = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=176,  # Must be divisible by 16
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

    #counter=0 #variabel yang nilainya nol, sebagai penanda itu frame ke berapa


    while True:
        if capture_device.stopped is True :
            break
        else :
            frame = capture_device.read()
        
        # counter += 1 #variabel counter +1
        # #ketika frame bernilai genap, maka di skip
        # if counter %8 != 0: #ketika counter jika dibagi dua nilainya tidak sama dengan nol
        #     continue #maka continue/iterasi ke loop berikutnya
        
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
        
        # morphology settings
        kernel = np.ones((4,4),np.uint8)
 
        counter = 400
 
        while counter < 650:
 
        # increment counter
            counter += 1
 
        # only process every third image (so as to speed up video)
            if counter % 3 != 0: continue

        # Convert disparity to depth
        depth = (focal_length * baseline) / (disparity + 1e-6)
        depth = cv.normalize(depth, depth , alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        depth  = np.uint8(depth)
        print(np.min(depth),np.max(depth))

        # Apply thresholding to the disparity map ______COBA GANTI ANGKA THRESHOLDNYA________
        threshold_value = 110 # Adjust this value as needed, 
        _, dip_thresholded = cv.threshold(disparity, threshold_value, 255, cv.THRESH_BINARY) #cari dikelas threshold kenapa dia ada 4 argumen
        dip_thresholded_resized = cv.resize(dip_thresholded, (512, 512))

        # apply morphological transformation
        opening = cv.morphologyEx(dip_thresholded_resized, cv.MORPH_CLOSE, kernel)
        opening1 = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
        opening2 = cv.morphologyEx(opening1, cv.MORPH_CLOSE, kernel)
        closing = cv.morphologyEx(opening2, cv.MORPH_OPEN, kernel)
        closing1 = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
        closing2 = cv.morphologyEx(closing1, cv.MORPH_OPEN, kernel)
        # morphology = cv.morphologyEx(dip_thresholded_resized, cv.MORPH_OPEN, kernel)
        # openclose = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
        # gradient = cv.morphologyEx(openclose, cv.MORPH_GRADIENT, kernel)

        #Dilatation for Background Implementation
        sure_bg = cv.dilate(closing2, kernel, iterations=3)

        

        # #DEPTH THRESHOLD
        threshold_value = 115 # Adjust this value as needed, 
        _, dep_thresholded = cv.threshold(depth, threshold_value, 255, cv.THRESH_BINARY) #cari dikelas threshold kenapa dia ada 4 argumen
        dep_thresholded_resized = cv.resize(dep_thresholded, (512, 512))
        # '''Threshold adalah nilai yang ditentukan untuk memberi batasan pada suatu pixel (segmentasi lah kurleb)'''

        # Apply bilateral filter to smooth the thresholded disparity map
        # dip_filtered = cv.bilateralFilter(dip_thresholded, d=9, sigmaColor=75, sigmaSpace=75) #cari dikelas bilateral 
        dip_filtered = cv.bilateralFilter(sure_bg, d=9, sigmaColor=75, sigmaSpace=75) #cari dikelas bilateral 
        dip_filtered = cv.resize(dip_filtered, (512, 512))
        threshold_filtered = cv.bilateralFilter(dip_thresholded, d=9, sigmaColor=75, sigmaSpace=75)

        # Apply a colormap to the filtered disparity map
        colormap_dip = cv.applyColorMap(dip_filtered, cv.COLORMAP_JET)
        colormap_threshold = cv.applyColorMap(threshold_filtered, cv.COLORMAP_JET)

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
        rd_colormap_dip = cv.resize(colormap_dip, (512, 512))
        colormap_threshold = cv.resize(colormap_threshold, (512, 512))
        # rd_dep = cv.resize(depth_normalized, (512, 512))

        # Create color bar with the same height as rd_colormap_dip
        color_bar = create_color_bar(cv.COLORMAP_JET, rd_colormap_dip.shape[0])#array ke shape 0, shape ini isinya ada 3 array

        # Combine color bar with the colored disparity image
        combined_display = cv.hconcat([rd_colormap_dip, color_bar])
        combined_display_Threshold = cv.hconcat([colormap_threshold, color_bar])

        # Add text to the color bar for legend
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(combined_display, 'Far', (rd_colormap_dip.shape[1] + 5, 20), font, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(combined_display, 'Near', (rd_colormap_dip.shape[1] + 5, combined_display.shape[0] - 10), font, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        # --- Masking Blue Color in Disparity Map ---

        # Convert the disparity image to HSV color space
        hsv = cv.cvtColor(rd_colormap_dip, cv.COLOR_BGR2HSV)
        hsv1 = cv.cvtColor(colormap_threshold, cv.COLOR_BGR2HSV)

        # Define range of blue color in HSV
        lower_blue = np.array([60, 40, 40])
        upper_blue = np.array([130, 255, 255])

        # Create a mask for the blue color
        mask = cv.inRange(hsv, lower_blue, upper_blue)

        # Invert the mask
        inverse_mask = cv.bitwise_not(mask)

        # Apply the mask to remove the blue regions
        # masked_resultr = cv.bitwise_and(rr_img, rr_img, mask=inverse_mask)
        masked_resultl = cv.bitwise_and(rl_img, rl_img, mask=inverse_mask)
        masked_left_resized = cv.resize(masked_resultl, (512, 512))
       
        left_midas  = compute_region_grayscale(masked_left_resized)
     
        left_midas = cv.normalize(left_midas, left_midas , alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        left_midas  = np.uint8(left_midas)
        print(np.min(left_midas),np.max(left_midas))

        # Apply thresholding to the midas map ______COBA GANTI ANGKA THRESHOLDNYA________
        threshold_value = 180  # Adjust this value as needed, 
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
        # masked_resultr = cv.bitwise_and(rr_img, rr_img, mask=inverse_maskmidas)
        masked_result_midas = cv.bitwise_and(rl_img, rl_img, mask=inverse_maskmidas)
        # masked_left_resized = cv.resize(masked_resultl, (512, 512))
        masked_left_midas = cv.resize(masked_result_midas, (512, 512))

       #membandingkan dimensi
        if len(rl_img.shape) == 2:  # If image is grayscale
            rl_img = cv.cvtColor(rl_img, cv.COLOR_GRAY2BGR)
        if len(dip_thresholded_resized.shape) == 2:#jika dia dimensinya 2, maka ditambahkan colorbgr jadilah 3d 
            dip_thresholded_resized = cv.cvtColor(dip_thresholded_resized, cv.COLOR_GRAY2BGR)
        if len(masked_left_resized.shape) == 2:
            masked_left_resized = cv.cvtColor(masked_left_resized, cv.COLOR_GRAY2BGR)       

        #combined the original file left, disparity, masking left
        combined_display_left = cv.hconcat([rl_img, dip_thresholded_resized, masked_left_resized])

       
        # --- Display Results ---

        #cv.imshow("Right", rr_img)
        # cv.imshow("Left", rl_img)
        # cv.imshow("ThresholdDisparity", dip_thresholded_resized)
        cv.imshow("closing", closing2)
        cv.imshow("bg", sure_bg)
        cv.imshow("Colored Closing with Legend", combined_display)  # Display the colored disparity map with legend
        # cv.imshow("Colored Disparity with Legend", combined_display_Threshold)
        #cv.imshow("Depth", rd_dep)
        # cv.imshow("Combined Left, Disparity, Masked Left", combined_display_left)
        # cv.imshow("masked midas", masked_midas)
        # cv.imshow("Disparity", rd_dip)
        # cv.imshow("morphology", morphology)
        # cv.imshow("openclose", openclose)
        # cv.imshow("Depth", dep_filtered)
    


        # Display the masked disparity map
        #cv.imshow('Masked right', masked_resultr)
        cv.imshow('masked left',masked_left_resized)
        cv.imshow('masked left midas',masked_left_midas)
        #cv.imshow('Original Frame', frame)          

        # Set mouse callback function to display disparity and depth
        # cv.setMouseCallback("Combined Left, Disparity, Masked Left", on_mouse_click)

        # Break the loop on 'q' key press
        if cv.waitKey(1) == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    capture_device.release()
    cv.destroyAllWindows()
