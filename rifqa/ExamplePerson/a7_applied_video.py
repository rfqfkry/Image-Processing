import numpy as np
import cv2 as cv

# split the image frames
def frame_splitter(source_image : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, src_w, _ = source_image.shape     # REFRENCE SHAPE (1440, 2880, 3)
    hwd = int(src_w // 2)                # LR (1440, 1440, 3)

    l_image = source_image[:, :hwd]
    r_image = source_image[:, hwd:]

    return l_image, r_image

def convert_to_grayscale(source_image : np.ndarray) -> np.ndarray:
    return cv.cvtColor(source_image, cv.COLOR_BGR2GRAY)

if __name__ == "__main__":

    capture_device = cv.VideoCapture(0)

    # Check if the camera opened successfully
    if not capture_device.isOpened():
        print("Error: Could not open camera.")
        exit()
    
    print("Starting, Press S to exit..")

    while True:
        # Capture frame-by-frame
        ret, frame = capture_device.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        left_img, right_img = frame_splitter(frame)

        # left_midas  = compute_region_grayscale(left_img)
        # right_midas = compute_region_grayscale(right_img)

        left_midas  = convert_to_grayscale(left_img)
        right_midas = convert_to_grayscale(right_img)

        # Step 2: Compute the disparity map
        stereo    = cv.StereoSGBM_create(
            numDisparities=144,  # Must be divisible by 16
            blockSize=7,
            uniquenessRatio=1,
        )
        disparity = stereo.compute(left_midas, right_midas)

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

        # Step 4: Apply the mask to the original image
        result_segmented = cv.bitwise_and(left_img, left_img, mask=disparity_smoothed.astype(np.uint8))

        # Display results
        result_segmented = cv.resize(result_segmented, (512, 512))
        cv.imshow('Result', result_segmented)

        # Break the loop on 'q' key press
        if cv.waitKey(1) == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    capture_device.release()
    cv.destroyAllWindows()


