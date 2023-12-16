import cv2
import numpy as np
import time
from mpi4py import MPI
from wm_utils import dct_2d, idct_2d, logistic_map_encryption

# 初始化一些变量
alpha = 0.1
block_size = 8
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
watermark_image_path = 'seu_logo_bw.jpg'
watermark_image = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)
watermark_image = cv2.resize(watermark_image, (256, 256), interpolation=cv2.INTER_AREA)
encrypted_watermark = logistic_map_encryption(watermark_image, key=0.5)
encrypted_watermark_flat = encrypted_watermark.flatten()

# 补0使长度为11的倍数
padding_length = fill_cnt - (len(encrypted_watermark_flat) % fill_cnt)
encrypted_watermark_flat = np.pad(encrypted_watermark_flat, (0, padding_length), 'constant')

# 加载宿主图像
host_image_path = 'frame.jpg'
bgr_image = cv2.imread(host_image_path)
yuv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YUV)
Y, U, V = cv2.split(yuv_image)

Y_shape_0=Y.shape[0]

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
np.set_printoptions(threshold=np.inf)

if rank == 0:
    start_time = time.time()

    # # 初始化一些变量
    # alpha = 0.1
    # block_size = 8
    # mask = np.array([
    #     [0, 0, 0, 0, 1, 1, 0, 0],
    #     [0, 0, 0, 1, 1, 0, 0, 0],
    #     [0, 0, 1, 1, 0, 0, 0, 0],
    #     [0, 1, 1, 0, 0, 0, 0, 0],
    #     [1, 1, 0, 0, 0, 0, 0, 0],
    #     [1, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0]
    # ])

    # # 计算mask中填充位数
    # fill_cnt = np.sum(mask)

    # 加载宿主图像
    host_image_path = 'frame.jpg'
    bgr_image = cv2.imread(host_image_path)
    yuv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(yuv_image)

    Y_shape_0=Y.shape[0]

    # # 加载并处理水印图像
    # watermark_image_path = 'seu_logo_bw.jpg'
    # watermark_image = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)
    # watermark_image = cv2.resize(watermark_image, (256, 256), interpolation=cv2.INTER_AREA)
    # encrypted_watermark = logistic_map_encryption(watermark_image, key=0.5)
    # encrypted_watermark_flat = encrypted_watermark.flatten()

    # # 补0使长度为11的倍数
    # padding_length = fill_cnt - (len(encrypted_watermark_flat) % fill_cnt)
    # encrypted_watermark_flat = np.pad(encrypted_watermark_flat, (0, padding_length), 'constant')


else:
    # alpha = None
    # block_size = None
    # fill_cnt = None
    # mask = None
    # encrypted_watermark_flat = None
    Y_shape_0 = None

# 广播一些共同的变量
# alpha = comm.bcast(alpha, root=0)
# block_size = comm.bcast(block_size, root=0)
# fill_cnt = comm.bcast(fill_cnt, root=0)
# mask = comm.bcast(mask, root=0)
# encrypted_watermark_flat = comm.bcast(encrypted_watermark_flat, root=0)
Y_shape_0 = comm.bcast(Y_shape_0, root=0)

if rank == 0:
    # 主进程分发块
    start_part_time=time.time()
    for i in range(0, Y.shape[0], block_size):
        for j in range(0, Y.shape[1], block_size):
            block = Y[i:i+block_size, j:j+block_size]
            dest_rank = (i // block_size) * (Y.shape[1] // block_size) + (j // block_size)
            #dest_rank = (i // block_size) * (Y.shape[0] // block_size) + (j // block_size)
            dest_rank = dest_rank % (size - 1) + 1
            comm.send((block, i, j), dest=dest_rank)

    # 主进程收集处理后的块
    for _ in range((Y.shape[0] // block_size) * (Y.shape[1] // block_size)):
        processed_block, i, j = comm.recv(source=MPI.ANY_SOURCE)
        Y[i:i+block_size, j:j+block_size] = processed_block

    # 发送结束信号给所有工作进程
    for r in range(1, size):
        comm.send((None, None, None), dest=r)

else:
    # 工作进程处理块
    while True:
        block, i, j = comm.recv(source=0)
        if block is None:  # 收到结束信号
            break

        index = (((i // block_size) * (Y_shape_0// block_size) + (j // block_size)) * fill_cnt) % len(encrypted_watermark_flat)
        dct_block = dct_2d(block)
        for x in range(block_size):
            for y in range(block_size):
                if mask[x, y] == 1 and index < len(encrypted_watermark_flat):
                    dct_block[x, y] += alpha * encrypted_watermark_flat[index]
                    index += 1
        idct_block = idct_2d(dct_block)
        comm.send((idct_block, i, j), dest=0)


if rank == 0:
    end_part_time=time.time()
    print("分块处理运行时间：", end_part_time - start_part_time, "秒")
    # 合并处理后的Y通道和原始的U、V通道，并保存含水印图像
    merged_yuv_image = cv2.merge([Y, U, V])
    reconstructed_bgr_image = cv2.cvtColor(merged_yuv_image, cv2.COLOR_YUV2BGR)
    cv2.imwrite('embedded_image_mpi.jpg', reconstructed_bgr_image)

    # 打印总运行时间
    total_time = time.time() - start_time
    print("总运行时间：", total_time, "秒")
