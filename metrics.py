import numpy as np
import cv2

def MSE(original, modified):
    diff = modified.astype(float) - original.astype(float)
    return np.mean(diff**2)

def PSNR(original, modified):
    mse = MSE(original, modified)
    if mse < 1e-10:
        return float('inf')
    return 20*np.log10(np.iinfo(modified.dtype).max)-10*np.log10(mse)

def gaussian_kernel(size, sigma=1.5):
    k = cv2.getGaussianKernel(size, sigma)
    kernel = k @ k.T
    return kernel / kernel.sum()

def SSIM_windowed(original, modified, window_size=11, k1=0.01, k2=0.03, a=1, b=1, y=1):
    mod_gray = cv2.cvtColor(modified, cv2.COLOR_BGR2GRAY).astype(float)
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(float)
    
    L = 255.0
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    c3 = c2 / 2.0
    
    h, w = mod_gray.shape
    half = window_size // 2
    ssim_map = np.zeros((h, w))

    mod_padded = np.pad(mod_gray, half, mode='reflect')
    orig_padded = np.pad(orig_gray, half, mode='reflect')

    for py in range(h):
        for px in range(w):
            patch_mod = mod_padded[py:py+window_size, px:px+window_size]
            patch_orig = orig_padded[py:py+window_size, px:px+window_size]

            kernel = gaussian_kernel(window_size)
            
            mu_x = np.sum(kernel * patch_mod)
            mu_y = np.sum(kernel* patch_orig)
            sigma_x = np.sqrt(np.sum(kernel * (patch_mod - mu_x)**2))
            sigma_y = np.sqrt(np.sum(kernel * (patch_orig - mu_y)**2))
            cov = np.sum(kernel * (patch_mod - mu_x) * (patch_orig - mu_y))
            
            l = (2*mu_x*mu_y + c1) / (mu_x**2 + mu_y**2 + c1)
            c_ = (2*sigma_x*sigma_y + c2) / (sigma_x**2 + sigma_y**2 + c2)
            s = (cov + c3) / (sigma_x*sigma_y + c3)
            
            ssim_map[py, px] = (l**a) * (c_**b) * (s**y)
    
    return np.mean(ssim_map), ssim_map


def SSIM(original, modified, k1 = 0.01, k2 = 0.03, a = 1, b = 1, y = 1):
    L = np.iinfo(modified.dtype).max

    mu_x = np.mean(modified)
    sigma_x = np.std(modified)

    mu_y = np.mean(original)
    sigma_y = np.std(original)

    cov_matrix = np.cov(modified.flatten(), original.flatten())
    cov = cov_matrix[0,1]  

    c1 = (k1*L)**2
    l = (2*mu_x*mu_y+c1)/(mu_x**2 + mu_y**2 +c1)

    c2 = (k2*L)**2
    c = (2*sigma_x*sigma_y + c2)/(sigma_x**2+sigma_y**2+c2)

    c3 = c2/2.0
    s = (cov + c3)/(sigma_x*sigma_y+c3)

    return (l**a)*(c**b)*(s**y)



from scipy.ndimage import uniform_filter, gaussian_filter

def SSIM_windowed_fast(original, modified, window_size=11, sigma=1.5, k1=0.01, k2=0.03, a=1, b=1, y=1):
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(float)
    mod_gray  = cv2.cvtColor(modified, cv2.COLOR_BGR2GRAY).astype(float)

    L  = 255.0
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    c3 = c2 / 2.0

    # windowed means via gaussian filter (replaces loop + np.mean per patch)
    mu_x = gaussian_filter(orig_gray, sigma=sigma)
    mu_y = gaussian_filter(mod_gray,  sigma=sigma)

    # windowed variances
    sigma_x = np.sqrt(np.maximum(gaussian_filter(orig_gray**2, sigma=sigma) - mu_x**2, 0))
    sigma_y = np.sqrt(np.maximum(gaussian_filter(mod_gray**2,  sigma=sigma) - mu_y**2, 0))

    # windowed covariance
    cov = gaussian_filter(orig_gray * mod_gray, sigma=sigma) - mu_x * mu_y

    # ssim components per pixel
    l = (2 * mu_x * mu_y + c1)       / (mu_x**2 + mu_y**2 + c1)
    c_ = (2 * sigma_x * sigma_y + c2) / (sigma_x**2 + sigma_y**2 + c2)
    s = (cov + c3)                    / (sigma_x * sigma_y + c3)

    ssim_map = (l**a) * (c_**b) * (s**y)

    return float(np.mean(ssim_map)), ssim_map
