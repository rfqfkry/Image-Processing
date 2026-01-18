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

    # capture_device = WebcamStream(stream_id=0) #  stream_id = 0 is for primary camera 
    # capture_device.start()

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

    print("Starting, Press Q to exit..")

    counter=0 #variabel yang nilainya nol, sebagai penanda itu frame ke berapa

    prev_time = time.time()

    #jarak objek ke margin
    margin = 10

    while True:
        # if capture_device.stopped is True :
        #     break
        # else :
        jhkl, frame=capture_device.read()
        # frame = capture_device.read()
        
        counter += 1 #variabel counter +1
        #ketika frame bernilai genap, maka di skip
        if counter % 5 != 0: #ketika counter jika dibagi dua nilainya tidak sama dengan nol
            continue #maka continue/iterasi ke loop berikutnya
        
        print("Counter frame:",counter)
        
        l_img, r_img = frame_splitter(frame)


        gl_img = cv.cvtColor(l_img, cv.COLOR_BGR2GRAY)
        gr_img = cv.cvtColor(r_img, cv.COLOR_BGR2GRAY)

        # Compute the disparity map with more detail
        disparity = stereo.compute(gl_img, gr_img)
        
        # Normalize the disparity map for visualization
        disparity = cv.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        disparity = np.uint8(disparity)
        
        # morphology settings
        kernel = np.ones((4,4),np.uint8)
 

        # Apply thresholding to the disparity map ______COBA GANTI ANGKA THRESHOLDNYA________
        threshold_value = 110 # Adjust this value as needed, 
        _, dip_thresholded = cv.threshold(disparity, threshold_value, 255, cv.THRESH_BINARY) #cari dikelas threshold kenapa dia ada 4 argumen
        dip_thresholded_resized = cv.resize(dip_thresholded, (256, 256))

        # apply morphological transformation
        opening = cv.morphologyEx(dip_thresholded_resized, cv.MORPH_CLOSE, kernel, iterations=3)
        closing2 = cv.morphologyEx(opening, cv.MORPH_OPEN, kernel, iterations=3)

        #Dilatation for Background Implementation
        sure_bg = cv.dilate(closing2, kernel, iterations=3)

        #find contours from the binary mask
        contours, _ = cv.findContours(sure_bg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # If no contours found, return the original image
        if contours:
            # Get the bounding box of the largest contour (which should be the object)
            x, y, w, h = cv.boundingRect(contours[0])
    
            # Add the margin, ensuring the values remain within the image boundaries
            x_margin = max(0, x - margin)
            y_margin = max(0, y - margin)
            w_margin = min(l_img.shape[1], x + w + margin) - x_margin
            h_margin = min(l_img.shape[0], y + h + margin) - y_margin
        
        # Apply bilateral filter to smooth the thresholded disparity map
        dip_filtered = cv.bilateralFilter(sure_bg, d=9, sigmaColor=75, sigmaSpace=75) #cari dikelas bilateral 
        dip_filtered = cv.resize(dip_filtered, (256, 256))

        # Apply a colormap to the filtered disparity map
        colormap_dip = cv.applyColorMap(dip_filtered, cv.COLORMAP_JET)

        # Update global variables
        disparity_map = disparity

        rl_img = cv.resize(l_img, (256, 256))

        rd_colormap_dip = cv.resize(colormap_dip, (256, 256))

        # Create color bar with the same height as rd_colormap_dip
        color_bar = create_color_bar(cv.COLORMAP_JET, rd_colormap_dip.shape[0])#array ke shape 0, shape ini isinya ada 3 array

        # Combine color bar with the colored disparity image
        combined_display = cv.hconcat([rd_colormap_dip, color_bar])

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
        masked_resultl = cv.bitwise_and(rl_img, rl_img, mask=inverse_mask)
        masked_left_resized = cv.resize(masked_resultl, (256, 256))
       
        left_midas  = compute_region_grayscale(masked_left_resized)
     
        left_midas = cv.normalize(left_midas, left_midas , alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        left_midas  = np.uint8(left_midas)

        # Apply thresholding to the midas map ______COBA GANTI ANGKA THRESHOLDNYA________
        threshold_value = 180  # Adjust this value as needed, 
        _, midas_thresholded = cv.threshold(left_midas, threshold_value, 255, cv.THRESH_BINARY) #cari dikelas threshold kenapa dia ada 4 argumen
        midas_thresholded_resized = cv.resize(midas_thresholded, (256, 256))
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
        # masked_left_resized = cv.resize(masked_resultl, (256, 256))
        masked_left_midas = cv.resize(masked_result_midas, (256, 256))

        # Crop the image with the calculated margins
        cropped_image = masked_left_midas[y_margin:y_margin + h_margin, x_margin:x_margin + w_margin]
    
        # return cropped_image
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
        # cv.imshow("closing", closing2)
        # cv.imshow("bg", sure_bg)
        # cv.imshow("Colored Closing with Legend", combined_display)  # Display the colored disparity map with legend
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
        cv.imshow('Cropped Frame', cropped_image)          

        # Set mouse callback function to display disparity and depth
        # cv.setMouseCallback("Combined Left, Disparity, Masked Left", on_mouse_click)

        current_time = time.time()
        interval_time = current_time - prev_time
        prev_time = current_time
        print(f"Interval antar frame: {interval_time * 1000:.2f} ms")

        # Break the loop on 'q' key press
        if cv.waitKey(1) == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    capture_device.release()
    cv.destroyAllWindows()
