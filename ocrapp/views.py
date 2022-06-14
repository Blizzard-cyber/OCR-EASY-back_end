import tempfile
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.core import serializers
import json
from aip import AipOcr  # 百度OCR SDK
from .models import Recon
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# Create your views here.
IMAGE_SIZE = 1800
BINARY_THREHOLD = 180

"""定义常量"""
APP_ID = '26018539'
API_KEY = 'GhxFjCqPGKMevLKg5HAaHc2b'
SECRET_KEY = 'wmdK8HBWZLCgn42FYaGlxq2DubjigupS'

"""初始化对象"""
client = AipOcr(APP_ID, API_KEY, SECRET_KEY)


# class rcoViewSet(viewsets.ModelViewSet):
#     queryset = Recon.objects.all()
#     serializer_class = rcoSerializer


@require_http_methods(["POST"])
def ocr_rco(request):
    img = request.FILES.get('file')  # 获取图片 type:InMemoryUploadedFile
    processedImg = img_process(img)  # 图片处理 传入File对象  输出处理后的图片二进制，将处理后的图片写盘

    # 从buffer目录中将图片读入内存并将类型强制转化为InMemoryUploadedFile
    gray_img = uploadfiles('./buffer/gray_photo.jpg', 'gray_photo')
    binary_img = uploadfiles('./buffer/binary_photo.jpg', 'binary_photo')
    morpho_img = uploadfiles('./buffer/morpho_photo.jpg', 'morpho_photo')
    smoothed_img = uploadfiles('./buffer/smoothed_photo.jpg', 'smoothed_photo')
    final_img = uploadfiles('./buffer/final_photo.jpg', 'final_photo')
    response = {}
    contex = []
    try:
        """调用通用文字识别接口, 识别本地图像"""
        result = client.basicGeneral(processedImg[0])  # 图像识别 传入图片二进制 输出识别结果json数组
        print(result)
        # for i in range(0, len(result['words_result'])):
        #     contex.append(result['words_result'][i]['words'])
        contex = result['words_result']
        Recon.objects.create(ori_photo=img, gray_photo=gray_img, binary_photo=binary_img,
                             morpho_photo=morpho_img, smoothed_photo=smoothed_img,
                             final_photo=final_img, context=contex)  # 写入数据库
        response['msg'] = 'success'
        response['error_num'] = 0
        response['id'] = Recon.objects.latest('id').id  # 返回对应的id号
        response['data'] = contex  # 返回识别结果
    except Exception as e:
        response['msg'] = str(e)
        response['error_num'] = 1
    return JsonResponse(response)


@require_http_methods(["GET"])
def show_context(request):
    query_id = request.GET.get('ID')
    response = {}
    try:
        lists = Recon.objects.filter(id=query_id)  # 根据传入的id号查询数据库

        # 将查询结果转化为json格式封装到response中
        response['data'] = json.loads(serializers.serialize("json", lists))
        response['msg'] = 'success'
        response['error_num'] = 0
    except  Exception as e:
        response['msg'] = str(e)
        response['error_num'] = 1

    return JsonResponse(response)


def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = max(1, int(IMAGE_SIZE / length_x))
    size = factor * length_x, factor * width_y
    # size = (1800, 1800)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename


def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def remove_noise_and_smooth(file_name):
    gray_img = cv2.imread(file_name, 0)  # 以灰度图读入
    cv2.imwrite('./buffer/gray_photo.jpg', gray_img)  # 灰度图
    filtered = cv2.adaptiveThreshold(gray_img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,
                                     3)  # 自适应阈值二值化
    cv2.imwrite(r".\buffer\binary_photo.jpg", filtered)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)  # 形态学运算——先开后闭
    cv2.imwrite(r".\buffer\morpho_photo.jpg", closing)
    smooth_img = image_smoothening(gray_img)
    cv2.imwrite(r".\buffer\smoothed_photo.jpg", smooth_img)
    or_image = cv2.bitwise_or(smooth_img, closing)
    cv2.imwrite(r".\buffer\final_photo.jpg", or_image)
    return or_image


def get_file_content(filePath):
    with open(filePath, "rb") as fp:
        return fp


def arrtobiary(img_name):
    ret, buf = cv2.imencode(".jpg", img_name)
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
    # 写入原始图片到buffer目录
    with open('./buffer/ori_photo.jpg', 'wb') as fp:
        for chunk in img.chunks():
            fp.write(chunk)
    file_path = "./buffer/ori_photo.jpg"
    temp_filename = set_image_dpi(file_path)  # 修复图片DPI为300DPI
    im_new = remove_noise_and_smooth(temp_filename)  # 图像二值化并去噪平滑
    processed_img.append(arrtobiary(im_new))

    return processed_img
