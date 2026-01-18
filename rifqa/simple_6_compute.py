import cv2   as cv
import numpy as np

# Define the dimensions of the checkerboard
CHECKERBOARD = (6, 9)

# Termination criteria for corner sub-pixel accuracy
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points based on the checkerboard dimensions
objp        = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints  = []  # 3d points in real world space
imgpointsL = []  # 2d points in image plane for left camera
imgpointsR = []  # 2d points in image plane for right camera

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
    
    print("Starting, Press S to exit..")

    while True:
        # Capture frame-by-frame
        ret, frame = capture_device.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            exit()
        
        l_img, r_img = frame_splitter(frame)

        grayL = cv.cvtColor(l_img, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(r_img, cv.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        retL, cornersL = cv.findChessboardCorners(grayL, CHECKERBOARD, None)
        retR, cornersR = cv.findChessboardCorners(grayR, CHECKERBOARD, None)

        # If found, add object points, image points (after refining them)
        if retL and retR:
            objpoints.append(objp)

            corners2L = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            imgpointsL.append(corners2L)

            corners2R = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            imgpointsR.append(corners2R)

            # Draw and display the corners
            cv.drawChessboardCorners(l_img, CHECKERBOARD, corners2L, retL)
            cv.drawChessboardCorners(r_img, CHECKERBOARD, corners2R, retR)
        else:
            l_img = grayL # if not detected, show grayscale 
            r_img = grayR 
        r_img = cv.resize(r_img, (512, 512))
        l_img = cv.resize(l_img, (512, 512))

        total_detections = str(len(imgpointsR))
        cv.putText(l_img, total_detections, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(r_img, total_detections, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow("left",  l_img)
        cv.imshow("right", r_img)

        # Break the loop on 'q' key press
        if cv.waitKey(1) == ord('s'):
            break

    cv.destroyAllWindows()

    print("Computing Affinine Matrix")

    # Calibrate the stereo camera
    retL, mtxL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
    retR, mtxR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)

    #Stereo calibration
    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC

    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    retStereocalib, mtxL, distL, mtxR, distR, R, T, E, F = cv.stereoCalibrate(
        objpoints,
        imgpointsL,
        imgpointsR,
        mtxL,
        distL,
        mtxR,
        distR,
        grayL.shape[::-1],
        criteria_stereo,
        flags
    )

    # Save calibration results to numpy file
    np.savez('stereo_calibration.npz', mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R=R, T=T)

    print("Stereo Calibration Successful")
    print("Calibration data saved to stereo_calibration.npz")
    