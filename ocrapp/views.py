import math
import sys
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.core import serializers
import json
from aip import AipOcr  # 百度OCR SDK
from .models import Recon
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from PIL import Image
from io import BytesIO

# Create your views here.

"""定义常量"""
APP_ID = ''
API_KEY = ''
SECRET_KEY = ''

"""初始化对象"""
client = AipOcr(APP_ID, API_KEY, SECRET_KEY)


# class rcoViewSet(viewsets.ModelViewSet):
#     queryset = Recon.objects.all()
#     serializer_class = rcoSerializer


@require_http_methods(["POST"])
def ocr_rco(request):
    img = request.FILES.get('file')  # 获取图片 type:InMemoryUploadedFile
    processedImg = img_process(img)  # 图片处理 传入File对象  输出处理后的图片二进制，将处理后的图片写盘
    gray_img = uploadfiles('./buffer/gray_photo.jpg', 'gray_photo')  # 从buffer目录中将图片读入内存并将类型强制转化为InMemoryUploadedFile
    binary_img = uploadfiles('./buffer/binary_photo.jpg', 'binary_photo')
    blur_img = uploadfiles('./buffer/blur_photo.jpg', 'blur_photo')
    response = {}
    contex = []
    try:
        """调用通用文字识别接口, 识别本地图像"""
        result = client.basicGeneral(processedImg[0])  # 图像识别 传入图片二进制 输出识别结果json数组
        print(result)
        # for i in range(0, len(result['words_result'])):
        #     contex.append(result['words_result'][i]['words'])
        contex=result['words_result']
        Recon.objects.create(ori_photo=img, gray_photo=gray_img, binary_photo=binary_img,
                             blur_photo=blur_img, context=contex)  # 写入数据库
        response['msg'] = 'success'
        response['error_num'] = 0
        response['id'] = Recon.objects.latest('id').id
        response['data'] = contex
    except Exception as e:
        response['msg'] = str(e)
        response['error_num'] = 1
    return JsonResponse(response)


@require_http_methods(["GET"])
def show_context(request):
    query_id = request.GET.get('ID')
    response = {}
    try:
        lists = Recon.objects.filter(id = query_id)
        response['data'] = json.loads(serializers.serialize("json", lists))
        response['msg'] = 'success'
        response['error_num'] = 0
    except  Exception as e:
        response['msg'] = str(e)
        response['error_num'] = 1

    return JsonResponse(response)


# 自适应中值滤波
def auto_median_filter(image, max_size):
    origen = 3  # 初始窗口大小
    board = origen // 2  # 初始应扩充的边界
    # max_board = max_size//2                                         # 最大可扩充的边界
    copy = cv.copyMakeBorder(image, *[board] * 4, borderType=cv.BORDER_DEFAULT)  # 扩充边界
    out_img = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            def sub_func(src, size):  # 两个层次的子函数
                kernel = src[i:i + size, j:j + size]
                # print(kernel)
                z_med = np.median(kernel)
                z_max = np.max(kernel)
                z_min = np.min(kernel)
                if z_min < z_med < z_max:  # 层次A
                    if z_min < image[i][j] < z_max:  # 层次B
                        return image[i][j]
                    else:
                        return z_med
                else:
                    next_size = cv.copyMakeBorder(src, *[1] * 4, borderType=cv.BORDER_DEFAULT)  # 增尺寸
                    size = size + 2  # 奇数的核找中值才准确
                    if size <= max_size:
                        return sub_func(next_size, size)  # 重复层次A
                    else:
                        return z_med

            out_img[i][j] = sub_func(copy, origen)
    return out_img


def integral(img):
    '''
    计算图像的积分和平方积分
    :param img:Mat--- 输入待处理图像
    :return:integral_sum, integral_sqrt_sum：Mat--- 积分图和平方积分图
    '''
    integral_sum = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
    integral_sqrt_sum = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)

    rows, cols = img.shape
    for r in range(rows):
        sum = 0
        sqrt_sum = 0
        for c in range(cols):
            sum += img[r][c]
            sqrt_sum += math.sqrt(img[r][c])

            if r == 0:
                integral_sum[r][c] = sum
                integral_sqrt_sum[r][c] = sqrt_sum
            else:
                integral_sum[r][c] = sum + integral_sum[r - 1][c]
                integral_sqrt_sum[r][c] = sqrt_sum + integral_sqrt_sum[r - 1][c]

    return integral_sum, integral_sqrt_sum


def sauvola(img, k, kernerl):
    '''
    sauvola阈值法。
    根据当前像素点邻域内的灰度均值与标准方差来动态计算该像素点的阈值
    :param img:Mat--- 输入待处理图像
    :param k:float---修正参数,一般0<k<1
    :param kernerl:set---窗口大小
    :return:img:Mat---阈值处理后的图像
    '''
    if kernerl[0] % 2 != 1 or kernerl[1] % 2 != 1:
        raise ValueError('kernerl元组中的值必须为奇数,'
                         '请检查kernerl[0] or kernerl[1]是否为奇数!!!')

    # 计算积分图和积分平方和图
    integral_sum, integral_sqrt_sum = integral(img)
    # integral_sum, integral_sqrt_sum = cv2.integral2(img)
    # integral_sum=integral_sum[1:integral_sum.shape[0],1:integral_sum.shape[1]]
    # integral_sqrt_sum=integral_sqrt_sum[1:integral_sqrt_sum.shape[0],1:integral_sqrt_sum.shape[1]]

    # 创建图像
    rows, cols = img.shape
    diff = np.zeros((rows, cols), np.float32)
    sqrt_diff = np.zeros((rows, cols), np.float32)
    mean = np.zeros((rows, cols), np.float32)
    threshold = np.zeros((rows, cols), np.float32)
    std = np.zeros((rows, cols), np.float32)

    whalf = kernerl[0] >> 1  # 计算领域类半径的一半

    for row in range(rows):
        print('第{}行处理中...'.format(row))
        for col in range(cols):
            xmin = max(0, row - whalf)
            ymin = max(0, col - whalf)
            xmax = min(rows - 1, row + whalf)
            ymax = min(cols - 1, col + whalf)

            area = (xmax - xmin + 1) * (ymax - ymin + 1)
            if area <= 0:
                print("error")
                sys.exit(1)

            if xmin == 0 and ymin == 0:
                diff[row, col] = integral_sum[xmax, ymax]
                sqrt_diff[row, col] = integral_sqrt_sum[xmax, ymax]
            elif xmin > 0 and ymin == 0:
                diff[row, col] = integral_sum[xmax, ymax] - integral_sum[xmin - 1, ymax]
                sqrt_diff[row, col] = integral_sqrt_sum[xmax, ymax] - integral_sqrt_sum[xmin - 1, ymax]
            elif xmin == 0 and ymin > 0:
                diff[row, col] = integral_sum[xmax, ymax] - integral_sum[xmax, ymax - 1]
                sqrt_diff[row, col] = integral_sqrt_sum[xmax, ymax] - integral_sqrt_sum[xmax, ymax - 1]
            else:
                diagsum = integral_sum[xmax, ymax] + integral_sum[xmin - 1, ymin - 1]
                idiagsum = integral_sum[xmax, ymin - 1] + integral_sum[xmin - 1, ymax]
                diff[row, col] = diagsum - idiagsum

                sqdiagsum = integral_sqrt_sum[xmax, ymax] + integral_sqrt_sum[xmin - 1, ymin - 1]
                sqidiagsum = integral_sqrt_sum[xmax, ymin - 1] + integral_sqrt_sum[xmin - 1, ymax]
                sqrt_diff[row, col] = sqdiagsum - sqidiagsum

            mean[row, col] = diff[row, col] / area
            std[row, col] = math.sqrt((sqrt_diff[row, col] - math.sqrt(diff[row, col]) / area) / (area - 1))
            threshold[row, col] = mean[row, col] * (1 + k * ((std[row, col] / 128) - 1))

            if img[row, col] < threshold[row, col]:
                img[row, col] = 0
            else:
                img[row, col] = 255

    return img


def get_file_content(filePath):
    with open(filePath, "rb") as fp:
        return fp


def arrtobiary(img_name):
    ret, buf = cv.imencode(".jpg", img_name)
    img_bin = Image.fromarray(np.uint8(buf)).tobytes()
    return img_bin


def uploadfiles(filePath, filename):
    Imgs = Image.open(filePath)
    pic_io = BytesIO()
    Imgs.save(pic_io, Imgs.format)
    pic_file = InMemoryUploadedFile(
        file=pic_io,
        name=filename + '.jpg',
        size=Imgs.size,
        content_type=type(Imgs),
        field_name=None,
        charset=None
    )
    return pic_file


def img_process(img):
    processed_img = []
    with open('./buffer/ori_photo.jpg', 'wb') as fp:
        for chunk in img.chunks():
            fp.write(chunk)
    gray_img = cv.imread(r'./buffer/ori_photo.jpg', 0)
    cv.imwrite('./buffer/gray_photo.jpg', gray_img)
    binary_img = cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    cv.imwrite('./buffer/binary_photo.jpg', binary_img)
    blur_img = cv.medianBlur(binary_img, 5)
    cv.imwrite('./buffer/blur_photo.jpg', blur_img)
    processed_img.append(arrtobiary(blur_img))

    return processed_img
