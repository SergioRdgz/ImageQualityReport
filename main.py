import cv2
import numpy as np
from skimage.metrics import structural_similarity
from metrics import MSE, PSNR, SSIM_windowed

image_pairs = [
    ("images/image.jpg", "images/image-MSE-142-SSIM-0662.jpg"),
]

pairs = [
    (cv2.imread(original), cv2.imread(distorted))
    for original, distorted in image_pairs
]

def put_text(img, text, pos):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

for i, (original, distorted) in enumerate(pairs):

    # compute metrics
    mse_value = MSE(original, distorted)
    psnr_value = PSNR(original, distorted)
    my_score, my_map = SSIM_windowed(original, distorted)
    sk_score, sk_map = structural_similarity(original, distorted, channel_axis=2, full=True)

    # convert maps to heatmaps
    my_map_uint8 = (my_map * 255).astype(np.uint8)
    my_heatmap = cv2.applyColorMap(my_map_uint8, cv2.COLORMAP_JET)

    sk_map_uint8 = (sk_map * 255).astype(np.uint8)
    sk_heatmap = cv2.applyColorMap(sk_map_uint8, cv2.COLORMAP_JET)

    # fix skimage map shape if needed
    if sk_heatmap.shape != my_heatmap.shape:
        sk_map_gray = cv2.cvtColor(sk_map_uint8, cv2.COLOR_GRAY2BGR)
        sk_heatmap = cv2.applyColorMap(sk_map_gray, cv2.COLORMAP_JET)

    # labels on images
    orig_labeled = original.copy()
    dist_labeled = distorted.copy()

    put_text(orig_labeled, "Original", (10, 30))
    put_text(dist_labeled, f" PSNR: {psnr_value:.2f} SSIM :{my_score:.2f}", (10, 30))
    put_text(my_heatmap, f"My SSIM: {my_score:.4f}", (10, 30))
    put_text(sk_heatmap, f"skimage SSIM: {sk_score:.4f}", (10, 30))
    cv2.imshow("Original | Distorted", np.hstack((orig_labeled, dist_labeled)))
    cv2.imshow("My SSIM | skimage SSIM", np.hstack((my_heatmap, sk_heatmap)))
    
    #combined = np.hstack((orig_labeled, dist_labeled))
    #cv2.imshow(f"Pair {0+1}: original | distorted | my SSIM map | skimage SSIM map", combined)

cv2.waitKey(0)
cv2.destroyAllWindows()