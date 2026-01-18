import cv2

def find_camera_id():
    # Try camera IDs from 0 to 10
    for camera_id in range(10):
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            print(f"Camera found at ID: {camera_id}")
            cap.release()
        else:
            print(f"No camera at ID: {camera_id}")

    print("Finished checking camera IDs.")

find_camera_id()