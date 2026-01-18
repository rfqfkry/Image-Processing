'''This script show only an external cam'''

import cv2 as cv
import numpy as np

def list_cameras():
    index = 0
    arr = []
    while True:
        cap = cv.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

def detect_external_camera(cameras):
    # Identify the external camera by comparing resolution or using the index.
    for cam in cameras:
        cap = cv.VideoCapture(cam)
        if cap.isOpened():
            width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            if width > 640 or height > 480:
                # Assuming external camera has higher resolution.
                return cam
    return cameras[-1]  # Fallback: return the last camera if no external camera detected

def display_cameras(camera_indices):
    caps = [cv.VideoCapture(i) for i in camera_indices]
    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))  # Placeholder for unavailable camera

        combined_frame = np.hstack(frames)
        cv.imshow('Cameras', combined_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv.destroyAllWindows()

def main():
    cameras = list_cameras()
    if cameras:
        print("Available cameras:")
        for cam in cameras:
            print(f"Camera Index: {cam}")
        
        external_camera = detect_external_camera(cameras)
        print(f"Selected External Camera Index: {external_camera}")
        
        # Proceed with image processing using the selected external camera
        cap = cv.VideoCapture(external_camera)
        while True:
            ret, frame = cap.read()
            if ret:
                # Image processing logic here
                cv.imshow('External Camera Feed', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()
    else:
        print("No cameras detected.")

if __name__ == "__main__":
    main()
