import cv2
import numpy as np
from matplotlib import pyplot as plt

imgL = cv2.imread('./50cm.png',cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('./50cm1.png',cv2.IMREAD_GRAYSCALE)

if imgL is None or imgR is None:
    print("Error: One or both images not found or unable to load.")
    exit()

stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)
disparity = stereo.compute(imgL,imgR)
disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)

plt.imshow(disparity_normalized, cmap='gray')  # Use 'gray' for grayscale visualization
plt.title('Disparity Map')
plt.colorbar()  # Optional: adds a color bar to indicate the depth levels
plt.show()

'''plt.figure(figsize = (20,10))
plt.imshow(disparity,'disparity')
plt.xticks([])
plt.yticks([])'''
