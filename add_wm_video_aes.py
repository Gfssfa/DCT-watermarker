import cv2
import numpy as np
from wm_utils import dct_2d, idct_2d, logistic_map_encryption, calculate_psnr

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend


alpha = 0.05 # 水印添加强度
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
watermark_image_path = 'seu_logo_01.png'  # 替换为水印图像路径
watermark_image = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)
watermark_image = cv2.resize(watermark_image, (256, 256), interpolation=cv2.INTER_AREA)
encrypted_watermark = logistic_map_encryption(watermark_image, key=0.5)
# cv2.imwrite('encrypted_watermark.png', encrypted_watermark)
# 测试纯粹logistic加解密是否是完全可逆 ssim值是 1.0 
# decrypted_watermark = logistic_map_encryption(encrypted_watermark, key=0.5)
# cv2.imwrite('decrypted_watermark.png', decrypted_watermark)
# ssim = ssim(watermark_image, decrypted_watermark, data_range=decrypted_watermark.max() - decrypted_watermark.min())
# print(ssim)

key = b"Wat3rmark3r1sFun"
cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
encryptor = cipher.encryptor()

# 将图像数据转换为8位的一维数组
encrypted_watermark_flat = encrypted_watermark.flatten()
print(encrypted_watermark_flat.shape)
# 创建一个新数组用于存储加密后的像素值
encrypted_pixels = np.zeros_like(encrypted_watermark_flat, dtype=np.uint8)

# 每次处理16个像素（128位）
for i in range(0, len(encrypted_watermark_flat), 16):
    block = encrypted_watermark_flat[i:i+16]
    encrypted_block = encryptor.update(bytes(block))
    encrypted_pixels[i:i+16] = np.frombuffer(encrypted_block, dtype=np.uint8)
# print(encrypted_pixels[0:16])
# 验证水印图像AES-128 ECB模式可以正确解密
# #print(encrypted_pixels[0:16])
# # 创建一个新的Cipher实例用于解密
# cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
# decryptor = cipher.decryptor()

# # 创建一个数组用于存储解密后的像素值
# decrypted_pixels = np.zeros_like(encrypted_pixels, dtype=np.uint8)

# # 每次处理16个像素（128位）
# for i in range(0, len(encrypted_pixels), 16):
#     block = encrypted_pixels[i:i+16]
#     decrypted_block = decryptor.update(bytes(block))
#     decrypted_pixels[i:i+16] = np.frombuffer(decrypted_block, dtype=np.uint8)

# #print(decrypted_pixels[0:16])
# # 重塑为原始水印图像的尺寸（256x256）
# decrypted_aes_watermark = decrypted_pixels.reshape(256, 256)
# decrypted_watermark = logistic_map_encryption(decrypted_aes_watermark, key=0.5)
# ssim = ssim(watermark_image, decrypted_watermark, data_range=decrypted_watermark.max() - decrypted_watermark.min())
# #print(ssim)

encrypted_watermark_flat = encrypted_pixels

# 补0使长度为fill_cnt的倍数
padding_length = fill_cnt - (len(encrypted_watermark_flat) % fill_cnt)
encrypted_watermark_flat = np.pad(encrypted_watermark_flat, (0, padding_length), 'constant')
len_filled_watermark = len(encrypted_watermark_flat)

# 打开视频文件
cap = cv2.VideoCapture('video_4s.mp4')

# 确定视频编解码器和视频大小
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建视频写入对象
out = cv2.VideoWriter('watermarked_video.mp4', fourcc, fps, (width, height))

# 逐帧读取视频
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    yuv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)  # 修正颜色转换
    Y, U, V = cv2.split(yuv_image)
    index = 0
    
    for i in range(0, Y.shape[0], block_size):
        for j in range(0, Y.shape[1], block_size):
            block = Y[i:i+block_size, j:j+block_size]
            dct_block = dct_2d(block) #DCT变换

            for x in range(block_size):
                for y in range(block_size):
                    if mask[x, y] == 1:
                        dct_block[x, y] += alpha * encrypted_watermark_flat[index % len_filled_watermark]
                        dct_block[x, y] /= (1.0 + alpha)
                        index+=1

            # 逆DCT变换
            idct_block = idct_2d(dct_block)
            Y[i:i+block_size, j:j+block_size] = idct_block

    # 合并处理后的Y通道和原始的U、V通道，并保存含水印图像
    merged_yuv_image = cv2.merge([Y, U, V])
    reconstructed_frame = cv2.cvtColor(merged_yuv_image, cv2.COLOR_YUV2BGR)  # 修正颜色转换

    psnr_value = calculate_psnr(frame, reconstructed_frame)
    print("PSNR value is", psnr_value)
    # 写入处理后的帧
    out.write(reconstructed_frame)

# 释放资源
cap.release()
out.release()