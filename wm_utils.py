import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def logistic_map_encryption(image, key, r=3.99):
    """Apply logistic map encryption to an image."""
    encrypted_image = np.copy(image).astype(np.float32)  # 确保图像是浮点数类型
    x = key
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x = r * x * (1 - x)
            encrypted_image[i, j] = int(encrypted_image[i, j]) ^ int(x * 256)  # 转换为整数后执行异或
    return encrypted_image.astype(np.uint8)  # 转换回无符号整数类型

def dct_2d(block):
    """Perform 2D Discrete Cosine Transform on an 8x8 block."""
    return cv2.dct(np.float32(block))

def idct_2d(block):
    """Perform 2D Inverse Discrete Cosine Transform on an 8x8 block."""
    return cv2.idct(block)

def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


