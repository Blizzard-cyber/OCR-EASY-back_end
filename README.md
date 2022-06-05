# OCR EASY——面向OCR的噪音背景去除系统后端部分

> Keep study! Keep enthusiasm!
>
> [前端部分 Click Here](https://github.com/Blizzard-cyber/OCR-EASY-front_end)

## 项目介绍：

- ### 	前端传入本地图片文件

- ###      后端接受图片使用opencv进行图像处理

- ###      处理后的图像调用OCR API进行文字识别

- ###     将识别结果和处理后的图片全部保存在数据库

- ###    前端通过GET请求数据库的内容将结果回显到前端

## 技术栈：

- 脚手架：Vue-cli (Vue2)
- UI组件：Element-UI
- 后端框架：Django3
- 数据库：Mysql
- 接口测试：Postman
- OCR识别API：Baidu AipOcr
- 图像处理：OpenCV

## 食用指南

- [x] **食用之前一定先启动后端**！！！
- [x] **在setting.py中输入自己的Mysql账号密码**
- [x] **在views.py中输入接口的信息**

##  Setup

``` bash
# install dependencies
pip install -r requirements.txt

# fill in your Mysql  username and password

#  migrate your database
python manage.py makemigrations
python manage.py migrate

# run your django project
python manage.py runserver
```

For a detailed explanation on how things work, check out the [guide and docs](https://docs.djangoproject.com/en/3.1/) 
