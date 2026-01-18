
'''This script didn't show that 0 is webcam and 1 is a 3d cam'''

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
       # cv.imwrite("50cm.png", combined_frame)
        #exit()
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
            print(f"Camera index: {cam}")
        
        selected_cameras = input("Enter the indices of cameras to display (comma-separated): ")
        selected_indices = [int(i) for i in selected_cameras.split(',')]
        display_cameras(selected_indices)
    else:
        print("No cameras found.")

if __name__ == "__main__":
    main()