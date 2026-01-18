import numpy as np
import cv2 as cv
import torch
from scipy.ndimage import label
import matplotlib.pyplot as plt

device = torch.device('cpu')

def blur_image_on_mask(image: np.ndarray, mask: np.ndarray, blur_strength=(15, 15)) -> np.ndarray:
    mask = mask.astype(np.uint8) * 255
    blurred_image = cv.GaussianBlur(image, blur_strength, 0)
    inverse_mask = cv.bitwise_not(mask)
    result = cv.bitwise_and(image, image, mask=inverse_mask)
    result += cv.bitwise_and(blurred_image, blurred_image, mask=mask)
    return result

if __name__ == "__main__":
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas = midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # Mengakses webcam
    cap = cv.VideoCapture(0)  # 0 adalah default webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Tidak dapat mengambil frame dari webcam.")
            break

        # Konversi frame dari BGR ke RGB
        source_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        with torch.no_grad():
            input_batch: torch.Tensor = transform(source_image).to(device)
            prediction: torch.Tensor = midas(input_batch)

            prediction: torch.Tensor = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=source_image.shape[:2],
                mode="bicubic",
                align_corners=False,
            )

            prediction: torch.Tensor = torch.squeeze(prediction)

        output: np.ndarray = prediction.cpu().numpy()

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

            # Konversi kembali ke BGR untuk OpenCV
            resulting_image = cv.cvtColor(resulting_image, cv.COLOR_RGB2BGR)
        else:
            resulting_image = frame  # Jika tidak ada blob, tampilkan frame asli

        # Tampilkan hasilnya
        cv.imshow('Resulting Image', resulting_image)

        # Break loop jika 'q' ditekan
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Lepaskan webcam dan tutup jendela
    cap.release()
    cv.destroyAllWindows()
