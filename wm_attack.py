import cv2
import numpy as np
from wm_utils import dct_2d, idct_2d, logistic_map_encryption, calculate_psnr

def add_gaussian_noise(image, mean=0, std=1):
    gauss = np.random.normal(mean, std, image.size)
    gauss = gauss.reshape(image.shape).astype('uint8')
    noisy_image = cv2.add(image, gauss)
    return noisy_image

def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy_image = np.copy(image)
    salt = np.ceil(salt_prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255

    pepper = np.ceil(pepper_prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0
    return noisy_image

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def random_crop(image, crop_size):
    height, width = image.shape[:2]
    if height < crop_size[0] or width < crop_size[1]:
        return image

    # 计算随机裁剪区域的起始点
    x = np.random.randint(0, width - crop_size[1])
    y = np.random.randint(0, height - crop_size[0])

    # 创建一个与原始图像大小相同的遮罩
    mask = np.ones(image.shape[:2], dtype="uint8")

    # 将裁剪区域设为0（黑色）
    mask[y:y+crop_size[0], x:x+crop_size[1]] = 0

    # 应用遮罩
    cropped_image = cv2.bitwise_and(image, image, mask=mask)
    
    return cropped_image

def max_pooling(image, pool_size=2):
    pooled_image = cv2.resize(image, (0,0), fx=1/pool_size, fy=1/pool_size)
    pooled_image = cv2.resize(pooled_image, (image.shape[1], image.shape[0]))
    return pooled_image

# 示例使用
