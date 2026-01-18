import numpy as np
import cv2 as cv
import torch
from scipy.ndimage import label
import matplotlib.pyplot as plt

# Set device for MiDaS (CPU or GPU)
device = torch.device('cpu')

def list_cameras():
    """Detect available cameras."""
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

def load_midas_model(model_type="MiDaS_small"):
    """Load MiDaS model for depth estimation."""
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas = midas.to(device)
    midas.eval()

    # Load the necessary transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    
    return midas, transform

def blur_image_on_mask(image: np.ndarray, mask: np.ndarray, blur_strength=(15, 15)) -> np.ndarray:
    """Apply a blur to the image based on a mask."""
    mask = mask.astype(np.uint8) * 255
    blurred_image = cv.GaussianBlur(image, blur_strength, 0)
    inverse_mask = cv.bitwise_not(mask)
    result = cv.bitwise_and(image, image, mask=inverse_mask)
    result += cv.bitwise_and(blurred_image, blurred_image, mask=mask)
    return result

def process_frame_with_midas(frame, midas, transform):
    """Apply MiDaS depth estimation to a frame."""
    # Convert frame from BGR to RGB
    source_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    with torch.no_grad():
        input_batch = transform(source_image).to(device)
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=source_image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    percentile_pivot = np.percentile(output, 75)
    output[output < percentile_pivot] = np.min(output)
    output[output > percentile_pivot] = np.max(output)

    masking_binary_array = np.zeros_like(output)
    masking_binary_array[output == np.max(output)] = 1

    binary_mask = np.array(masking_binary_array, dtype=bool)
    labeled_array, num_features = label(binary_mask)
    
    if num_features > 0:
        largest_blob_label = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1
        largest_blob = (labeled_array == largest_blob_label)

        binary_mask = masking_binary_array
        background_mask = np.logical_not(masking_binary_array)

        resulting_image = blur_image_on_mask(source_image, background_mask, blur_strength=(257, 257))

        # Convert back to BGR for OpenCV
        resulting_image = cv.cvtColor(resulting_image, cv.COLOR_RGB2BGR)
    else:
        resulting_image = frame  # If no blob, display original frame

    return resulting_image

def main():
    # Load MiDaS model and transforms
    midas, transform = load_midas_model()

    # List available cameras
    cameras = list_cameras()
    if cameras:
        print("Available cameras:")
        for cam in cameras:
            print(f"Camera index: {cam}")
        
        # User selects the camera to use
        selected_camera = input("Enter the index of the external camera to use: ")
        selected_index = int(selected_camera)
        
        # Prompt for resizing to screen resolution
        resize_to_screen = input("Resize output to fit laptop screen? (y/n): ").strip().lower() == 'y'
        
        # Prompt for flipping the camera
        flip_camera = input("Flip camera horizontally? (y/n): ").strip().lower() == 'y'
        
        # Get screen resolution if resize is needed
        screen_width, screen_height = 1920, 1080  # Default Full HD
        if resize_to_screen:
            screen_resolution = input("Enter screen resolution in 'widthxheight' format (e.g., 1920x1080): ").strip()
            try:
                screen_width, screen_height = map(int, screen_resolution.split('x'))
            except ValueError:
                print("Invalid resolution format. Using default resolution (1920x1080).")

        # Open selected camera
        cap = cv.VideoCapture(selected_index)
        if not cap.isOpened():
            print(f"Failed to open camera with index {selected_index}.")
            return
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Cannot capture frame from camera.")
                break

            # Resize frame if needed
            if resize_to_screen:
                frame = cv.resize(frame, (screen_width, screen_height))
            
            # Flip frame if needed
            if flip_camera:
                frame = cv.flip(frame, 1)

            # Apply MiDaS processing to the frame
            processed_frame = process_frame_with_midas(frame, midas, transform)

            # Display the resulting frame
            cv.imshow('Resulting Image', processed_frame)

            # Break loop if 'q' is pressed
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera and close windows
        cap.release()
        cv.destroyAllWindows()
    else:
        print("No cameras found.")

if __name__ == "__main__":
    main()
