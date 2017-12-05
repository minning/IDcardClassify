# coding:utf8
'''
Author : minning
Date : 2017/10/13
代码目的：对图片进行重命名
          缩放到标准尺寸（320*240）

'''

import os
from PIL import Image
import numpy as np


ProcessedDataPath = '../PictureData'
folders = os.listdir(ProcessedDataPath)


i = 0
for folder in folders:
    folderAbsPath = os.path.join(ProcessedDataPath,folder)
    for file in os.listdir(folderAbsPath):
        fileAbspath = os.path.join(folderAbsPath,file)
        # 照片缩放到统一大小
        img = Image.open(fileAbspath)  # 读取和代码处于同一目录下的 lena.png
        width, height = 214, 135  # 用PIL进行尺寸修改,因为后面要计算总的 像素值，所以这里设置的小一点。
        out = img.resize((width, height), Image.ANTIALIAS)  # resize image with high-quality
        print(np.array(img).shape)
        # print(np.array(out).shape)
        print(out.getpixel((0,0)))
        type = 'png'
        out.save(fileAbspath, type)
        fileList = file.split('.')
        filename = fileList[0]
        filestyle = fileList[1]

        if filestyle =='png' or 'jpg':
            newname = str(i)+'.'+filestyle
            print (os.path.join(folderAbsPath, newname))
            if os.path.exists(os.path.join(folderAbsPath, newname)):
                pass
            else:
                # os.mkdir(os.path.join(folderAbsPath, newname))
                os.rename(os.path.join(folderAbsPath, file), os.path.join(folderAbsPath, newname))
        i += 1
