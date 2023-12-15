import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from wm_utils import dct_2d, idct_2d, logistic_map_encryption

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

alpha = 0.05
block_size = 8
# 定义掩码矩阵
mask = np.array([
    [0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])

# 计算mask中填充位数
fill_cnt = np.sum(mask)

# 加载并处理水印图像
watermark_image_path = 'seu_logo_01.png'
watermark_image = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)
watermark_image = cv2.resize(watermark_image, (256, 256), interpolation=cv2.INTER_AREA)
encrypted_watermark = logistic_map_encryption(watermark_image, key=0.5)
# cv2.imwrite('encrypted_watermark.png', encrypted_watermark)
encrypted_watermark_flat = encrypted_watermark.flatten()
len_watermark_flat=len(encrypted_watermark_flat)

# 补0使长度为fill_cnt的倍数
padding_length = fill_cnt - (len(encrypted_watermark_flat) % fill_cnt)
encrypted_watermark_flat = np.pad(encrypted_watermark_flat, (0, padding_length), 'constant')
len_filled_watermark = len(encrypted_watermark_flat)

# 打开原始视频和含水印视频
cap_original = cv2.VideoCapture('video_4s.mp4')
cap_watermarked = cv2.VideoCapture('watermarked_video.mp4')

fps = cap_original.get(cv2.CAP_PROP_FPS)
print(fps)

second_frame_count = 0
second_best_similarity = -1
second_best_watermark = None

while cap_original.isOpened() and cap_watermarked.isOpened():
    ret_original, frame_original = cap_original.read()
    ret_watermarked, frame_watermarked = cap_watermarked.read()
    if not (ret_original and ret_watermarked):
        break

    # 加载原始宿主图像并转换为YUV
    host_yuv_image = cv2.cvtColor(frame_original, cv2.COLOR_BGR2YUV) 
    Y, _, _ = cv2.split(host_yuv_image)  # 提取原始宿主图像的Y分量

    # 加载含水印的图像并转换为YUV
    embedded_yuv_image = cv2.cvtColor(frame_watermarked, cv2.COLOR_BGR2YUV)
    embedded_Y, _, _ = cv2.split(embedded_yuv_image)  # 提取含水印图像的Y分量


    # 分别对含水印的Y分量和原始宿主图像的Y分量进行DCT变换
    dct_embedded_Y = np.zeros_like(Y, dtype=np.float32)
    dct_original_Y = np.zeros_like(Y, dtype=np.float32)

    for i in range(0, Y.shape[0], block_size):
        for j in range(0, Y.shape[1], block_size):
            original_block = Y[i:i+block_size, j:j+block_size]
            embedded_block = embedded_Y[i:i+block_size, j:j+block_size]

            dct_original_Y[i:i+block_size, j:j+block_size] = dct_2d(original_block)
            dct_embedded_Y[i:i+block_size, j:j+block_size] = dct_2d(embedded_block)

    # 计算DCT变换的差值
    difference_dct_Y = dct_embedded_Y*(1.0 + alpha) - dct_original_Y
    # difference_dct_Y = dct_embedded_Y - dct_original_Y

    # 提取水印
    extracted_watermark_flat = []
    for i in range(0, difference_dct_Y.shape[0], block_size):
        for j in range(0, difference_dct_Y.shape[1], block_size):
            block = difference_dct_Y[i:i+block_size, j:j+block_size]

            # 使用掩码矩阵指定位置提取水印
            for x in range(block_size):
                for y in range(block_size):
                    if mask[x, y] == 1:
                        extracted_watermark_flat.append(block[x, y] / alpha)

    # 将提取的水印数据转换为NumPy数组
    extracted_watermark_array = np.array(extracted_watermark_flat)
    extracted_watermark_array_int = extracted_watermark_array.astype(np.uint8)

    flat_original_watermark = watermark_image.flatten()

    frame_best_similarity = -1
    frame_best_watermark = None
    frame_best_index = -1

    key = b"Wat3rmark3r1sFun"

    for k in range(len(extracted_watermark_array) // len_filled_watermark):
        start_index = k * len_filled_watermark
        watermark_group = extracted_watermark_array_int[start_index:start_index + len_filled_watermark]
        watermark = watermark_group[ : - padding_length].reshape(256, 256)
        watermark_flat = watermark.flatten()

        cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_pixels = np.zeros(len_watermark_flat, dtype=np.uint8)

        # 每次处理16个像素（128位）
        for i in range(0, len_watermark_flat, 16):
            block = watermark_flat[i:i+16]
            decrypted_block = decryptor.update(bytes(block))
            decrypted_pixels[i:i+16] = np.frombuffer(decrypted_block, dtype=np.uint8)

        decrypted_aes_watermark = decrypted_pixels.reshape(256, 256)   
        watermark = logistic_map_encryption(decrypted_aes_watermark, key=0.5)

        threshold = 127
        (thresh, watermark) = cv2.threshold(watermark, threshold, 255, cv2.THRESH_BINARY)

        frame_current_ssim = ssim(watermark_image, watermark, data_range=watermark.max() - watermark.min())
        print(frame_current_ssim)

        if frame_current_ssim > frame_best_similarity:
            frame_best_similarity = frame_current_ssim
            frame_best_watermark = watermark
            frame_best_index = k

    if frame_best_watermark is not None:
        #cv2.imwrite('best_extracted_watermark.jpg', best_watermark)
        print(f"Best extracted watermark index: {frame_best_index}")
        print(f"Best SSIM: {frame_best_similarity}")
    else:
        print("No watermark extracted.")


    second_frame_count += 1

    if frame_best_similarity > second_best_similarity :
         second_best_similarity = frame_best_similarity
         second_best_watermark = frame_best_watermark


    # 每秒保存一次SSIM最高的水印
    if second_frame_count == fps:
        if second_best_watermark is not None:
            cv2.imwrite(f'best_watermark_{int(cap_original.get(cv2.CAP_PROP_POS_FRAMES)) // fps}.png', second_best_watermark)
            print(f"Second Best SSIM: {second_best_similarity}")
            second_frame_count = 0
            second_best_similarity = -1
            second_best_watermark = None

cap_original.release()
cap_watermarked.release()
