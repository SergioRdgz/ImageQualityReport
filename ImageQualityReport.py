import cv2
import numpy as np

def MSE(modified, original):
    diff = modified.astype(float) - original.astype(float)
    return np.mean(diff**2)

img = cv2.imread("images/image.jpg")
img2 = img
img3 = cv2.imread("images/image-MSE-142-SSIM-0662.jpg")

mse_value = MSE(img, img2)

cv2.putText(img2, f"MSE: {mse_value:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

mse_value = MSE(img, img3)

from skimage.metrics import mean_squared_error

expected = mean_squared_error(img, img3)

print(f"mine:     {mse_value}")
print(f"skimage:  {expected}")

cv2.putText(img3, f"MSE: {mse_value:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)




combined = np.hstack((img, img2,img3))
cv2.imshow("original | copy | distorted", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()