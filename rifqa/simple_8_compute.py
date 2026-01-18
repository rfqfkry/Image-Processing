import cv2
import numpy as np

# Define the checkerboard dimensions
CHECKERBOARD = (6, 9)  # Adjust based on your checkerboard pattern

# Termination criteria for corner sub-pixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points
objpoints = []
imgpoints = []

# Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# split the image frames
def frame_splitter(source_image : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, src_w, _ = source_image.shape     # REFRENCE SHAPE (1440, 2880, 3)
    hwd = int(src_w // 2)                # LR (1440, 1440, 3)

    l_image = source_image[:, :hwd]
    r_image = source_image[:, hwd:]

    return l_image, r_image


if __name__ == "__main__":

    # Capture images for calibration
    cap = cv2.VideoCapture(1)

    print("press q to stop")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, _ = frame_splitter(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
        
        cv2.imshow('Calibration', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Calibrate the camera
    ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    np.save('./calibrate.np', 
        ret = ret,
        k = K, 
        d = D,
        rvecs = rvecs,
        tvecs = tvecs
    )