import numpy as np
import cv2 as cv
import torch
from scipy.ndimage import label
import matplotlib.pyplot as plt

device = torch.device('cpu')

def blur_image_on_mask(image : np.ndarray, mask : np.ndarray, blur_strength=(15, 15)) -> np.ndarray:
    """
    Blurs the areas of the image where the mask is True.

    Parameters:
    - image: The input image (numpy array).
    - mask: The 2D binary array (numpy array) with the same height and width as the image.
    - blur_strength: The strength of the blur (tuple of two integers).

    Returns:
    - Blurred image (numpy array).
    """
    # Ensure the mask is in the correct format
    mask = mask.astype(np.uint8) * 255

    # Create a blurred version of the image
    blurred_image = cv.GaussianBlur(image, blur_strength, 0)

    # Create an inverse mask
    inverse_mask = cv.bitwise_not(mask)

    # Combine the original image and the blurred image using the masks
    result  = cv.bitwise_and(image, image, mask=inverse_mask)
    result += cv.bitwise_and(blurred_image, blurred_image, mask=mask)

    return result


if __name__ == "__main__":
    print("---")

    model_type = "MiDaS_small"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas = midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    source_image = cv.imread('./webcam.jpg')
    source_image = cv.cvtColor(source_image, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(source_image)


    with torch.no_grad():
        input_batch : torch.Tensor = transform(source_image).to(device)
    
        prediction : torch.Tensor  = midas(input_batch)

        prediction : torch.Tensor  = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size          = source_image.shape[:2],
            mode          = "bicubic",
            align_corners = False,
        )

        prediction : torch.Tensor  = torch.squeeze(prediction)

    output : np.ndarray = prediction.cpu().numpy()

    plt.figure()
    plt.imshow(output)

    percentile_pivot = np.percentile(output, 75)

    output[output < percentile_pivot] = np.min(output)
    output[output > percentile_pivot] = np.max(output)

    plt.figure()
    plt.imshow(output)

    masking_binary_array = np.zeros_like(output)
    masking_binary_array[output == np.max(output)] = 1

    binary_mask     = np.array(masking_binary_array, dtype = bool)
    labeled_array, num_features = label(binary_mask)
    
    largest_blob_label = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1
    largest_blob       = (labeled_array == largest_blob_label)

    binary_mask     = masking_binary_array
    background_mask = np.logical_not(masking_binary_array) 

    plt.figure()
    plt.imshow(background_mask)

    resulting_image = blur_image_on_mask(source_image, background_mask, blur_strength = (257, 257))
   
    plt.figure()
    plt.imshow(resulting_image)

    plt.show()