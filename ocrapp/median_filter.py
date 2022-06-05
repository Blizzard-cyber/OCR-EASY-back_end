
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
# from functools import reduce               # 导入reduce，将一个函数作用在一个序列上，并且序列内容自动累计

# 显示汉字
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 定义坐标数字字体及大小
def label_def():
    plt.xticks(fontproperties='Times New Roman', size=8)
    plt.yticks(fontproperties='Times New Roman', size=8)


# 读取图片
img_saltpep = cv.imread('./out/gauss+salt.png', 0)               # 灰度图，噪声密度50%


def auto_median_filter(image, max_size):
    origen = 3                                                        # 初始窗口大小
    board = origen//2                                                 # 初始应扩充的边界
    # max_board = max_size//2                                         # 最大可扩充的边界
    copy = cv.copyMakeBorder(image, *[board]*4, borderType=cv.BORDER_DEFAULT)         # 扩充边界
    out_img = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            def sub_func(src, size):                         # 两个层次的子函数
                kernel = src[i:i+size, j:j+size]
                # print(kernel)
                z_med = np.median(kernel)
                z_max = np.max(kernel)
                z_min = np.min(kernel)
                if z_min < z_med < z_max:                                 # 层次A
                    if z_min < image[i][j] < z_max:                       # 层次B
                        return image[i][j]
                    else:
                        return z_med
                else:
                    next_size = cv.copyMakeBorder(src, *[1]*4, borderType=cv.BORDER_DEFAULT)   # 增尺寸
                    size = size+2                                        # 奇数的核找中值才准确
                    if size <= max_size:
                        return sub_func(next_size, size)     # 重复层次A
                    else:
                        return z_med
            out_img[i][j] = sub_func(copy, origen)
    return out_img


if __name__ == '__main__':                                       # 运行当前函数

    img_auto_filter = auto_median_filter(img_saltpep, 7)
    # img_median = cv.medianBlur(img_saltpep, 7)
    # img_re_median = cv.medianBlur(img_auto_filter, 3)                           # 两次3x3的中值滤波效果才能差不多
    plt.subplot(131), plt.imshow(img_saltpep, "gray"), plt.title('椒盐噪声密度50%', fontsize='small'), label_def()
    plt.subplot(132), plt.imshow(img_auto_filter, "gray"), plt.title('自适应中值滤波', fontsize='small'), label_def()
    # plt.subplot(133), plt.imshow(img_median, "gray"), plt.title('中值滤波', fontsize='small'), label_def()
    # plt.subplot(144), plt.imshow(img_re_median, "gray"), plt.title('再中值滤波', fontsize='small'), label_def()

    plt.show()
