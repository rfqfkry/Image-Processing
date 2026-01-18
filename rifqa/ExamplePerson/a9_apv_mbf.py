import threading
import cv2 as cv
import numpy as np
from multiprocessing import shared_memory

# Create shared memory for the output array
example_frame = np.random.random((1440, 1440)).astype(np.uint8)

# ouput buffer
shm_feed_output   = shared_memory.SharedMemory(create=True, size=example_frame.nbytes)
output_feed_latest = np.ndarray(example_frame.shape, dtype=example_frame.dtype, buffer=shm_feed_output.buf)

# input buffer
left_feed_latest  = np.zeros_like(example_frame)
right_feed_latest = np.zeros_like(example_frame)

# split the image frames
def frame_splitter(source_image : np.ndarray, elif_alif = False) -> tuple[np.ndarray, np.ndarray]:
    _, src_w, _ = source_image.shape     # REFRENCE SHAPE (1440, 2880, 3)
    hwd = int(src_w // 2)                # LR (1440, 1440, 3)

    l_image = source_image[:, :hwd]
    r_image = source_image[:, hwd:]

    if elif_alif:
        l_image = cv.resize(l_image, (1440, 1440))
        r_image = cv.resize(r_image, (1440, 1440))
    return l_image, r_image

def convert_to_grayscale(source_image : np.ndarray, elif_alif = False) -> np.ndarray:
    x = cv.cvtColor(source_image, cv.COLOR_BGR2GRAY)
    if elif_alif:
        x = cv.resize(x, (1440, 1440))
    return x

def compute_absolute_diffrance(stereo : cv.StereoBM, left_source : np.ndarray, right_source : np.ndarray) -> np.ndarray:
    disparity = stereo.compute(left_source, right_source)

    # Normalize the disparity for better visualization
    disparity_normalized = (disparity - np.min(disparity)) / np.ptp(disparity)
    disparity_normalized = disparity_normalized * 255
    disparity_normalized = disparity_normalized.astype(np.uint8)

    # create boundry
    lower_limits = 135
    upper_limits = 255

    masked_image = np.where(
        (disparity_normalized >= lower_limits) & (disparity_normalized <= upper_limits), 
        1, 
        0
    )

    masked_image = masked_image.astype(np.uint8)
    masked_image = masked_image * 255

    masked_image = cv.resize(masked_image, (512, 512))
    
    # Step 3: Apply post-processing to smooth the disparity map
    kernel = np.ones( (5, 5), np.uint8)
    disparity_smoothed = cv.medianBlur(masked_image, 9)
    disparity_smoothed = cv.erode(disparity_smoothed, kernel= kernel, iterations= 2)

    disparity_smoothed = cv.morphologyEx(disparity_smoothed, cv.MORPH_DILATE, kernel, iterations = 4)
    disparity_smoothed = cv.morphologyEx(disparity_smoothed, cv.MORPH_CLOSE,  kernel, iterations = 10)

    disparity_smoothed = cv.resize(disparity_smoothed, (1440, 1440))
    disparity_smoothed = np.where(
        (disparity_smoothed >= 100), 
        1, 
        0
    )
    disparity_smoothed = disparity_smoothed.astype(np.uint8)

    return disparity_smoothed

def background_task(left_source_latest : np.ndarray, right_source_latest : np.ndarray, output_feed_latest) -> None:
    stereo    = cv.StereoSGBM_create(
        numDisparities=144,  # Must be divisible by 16
        blockSize=7,
        uniquenessRatio=1,
    )
    while True:
        # actually process the feed here !
        diff_mask = compute_absolute_diffrance(stereo, left_source_latest, right_source_latest)
        np.copyto(output_feed_latest, diff_mask)
    print("Background Task Exitting...")
        

if __name__ == "__main__":

    capture_device = cv.VideoCapture(0)

    # Check if the camera opened successfully
    if not capture_device.isOpened():
        print("Error: Could not open camera.")
        exit()
    
    print("Starting, Press Q to exit..")

    # Start the background thread
    thread = threading.Thread(target=background_task, args=(left_feed_latest, right_feed_latest, output_feed_latest))
    thread.daemon = True
    thread.start()

    while True:
        # Capture frame-by-frame
        ret, frame = capture_device.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        left_img, right_img = frame_splitter(frame, elif_alif = True)

        left_midas  = convert_to_grayscale(left_img,  elif_alif = True)
        right_midas = convert_to_grayscale(right_img, elif_alif = True)

        # Step 2: Update the Frame Feed !
        left_feed_latest[:]  = left_midas
        right_feed_latest[:] = right_feed_latest

        # Step 4: Apply the mask to the original image using the latest mask !
        # print(output_feed_latest.shape, left_img.shape, left_img.dtype, output_feed_latest.dtype)
        result_segmented = cv.bitwise_and(left_img, left_img, mask=output_feed_latest)

        # Display results
        result_segmented = cv.resize(result_segmented, (512, 512))
        cv.imshow('Result', result_segmented)

        # Break the loop on 'q' key press
        if cv.waitKey(1) == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    capture_device.release()
    cv.destroyAllWindows()


