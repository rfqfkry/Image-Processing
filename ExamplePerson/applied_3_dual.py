import cv2   as cv
import numpy as np

# split the image frames
def frame_splitter(source_image : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    
    print("Starting, Press Q to exit..")

    while True:
        # Capture frame-by-frame
        ret, frame = capture_device.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break
        
        l_img, r_img = frame_splitter(frame)

        r_img = cv.resize(r_img, (512, 512))
        l_img = cv.resize(l_img, (512, 512))

        cv.imshow("left",  l_img)
        cv.imshow("right", r_img)

        # Break the loop on 'q' key press
        if cv.waitKey(1) == ord('q'):
            break

   # Release the camera and close all OpenCV windows
    capture_device.release()
    cv.destroyAllWindows()
