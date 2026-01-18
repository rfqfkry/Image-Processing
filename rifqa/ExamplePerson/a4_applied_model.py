import numpy as np
import cv2 as cv
import torch

device     = torch.device('cpu')

MIDAS_MODEL = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
MIDAS_MODEL.eval()

MIDAS_TRANFORM = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform


# split the image frames
def frame_splitter(source_image : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, src_w, _ = source_image.shape     # REFRENCE SHAPE (1440, 2880, 3)
    hwd = int(src_w // 2)                # LR (1440, 1440, 3)

    l_image = source_image[:, :hwd]
    r_image = source_image[:, hwd:]

    return l_image, r_image

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

    base_input_frame =  cv.imread('./images/example_50cm.png')

    left_img, right_img = frame_splitter(base_input_frame)

    left_midas  = compute_region_grayscale(left_img)
    right_midas = compute_region_grayscale(right_img)

    # Step 2: Compute the disparity map
    stereo    = cv.StereoBM_create(numDisparities=16, blockSize=5)
    disparity = stereo.compute(left_midas, right_midas)

    # Normalize the disparity for better visualization
    disparity_normalized = (disparity - np.min(disparity)) / np.ptp(disparity)
    disparity_normalized = disparity_normalized * 255
    disparity_normalized = disparity_normalized.astype(np.uint8)

    # Step 3: Apply post-processing to smooth the disparity map
    kernel = np.ones((5,5),np.uint8)
    disparity_smoothed = cv.morphologyEx(disparity_normalized, cv.MORPH_CLOSE, kernel)
    disparity_smoothed = cv.medianBlur(disparity_smoothed, 5)

    # Step 4: Threshold the smoothed disparity map to remove background
    threshold = 25  # Adjust this value to control how much background to remove
    _, mask = cv.threshold(disparity_smoothed, threshold, 255, cv.THRESH_BINARY)

    # Step 5: Further process the mask to get larger coherent regions
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)


    # Step 4: Apply the mask to the original image
    result = cv.bitwise_and(left_img, left_img, mask=mask.astype(np.uint8))

    # Display results
    left_img             = cv.resize(left_img, (512, 512))
    disparity_normalized = cv.resize(disparity_normalized, (512, 512))
    result               = cv.resize(result, (512, 512))

    left_midas  = cv.resize(left_midas,  (512, 512))
    right_midas = cv.resize(right_midas, (512, 512))    
    mask = cv.resize(mask, (512, 512))    

    cv.imshow('Original Left Image', left_img)
    cv.imshow('Disparity Map', disparity_normalized)
    cv.imshow('Result (Foreground Only)', result)
    cv.imshow('Result (left midas)', left_midas)
    cv.imshow('Result (right midas)', right_midas)
    cv.imshow('Result (mask)', mask)
    cv.waitKey(0)
    cv.destroyAllWindows()

