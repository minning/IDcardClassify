#!/user/bin/env python
# coding:utf-8
'''
    author : minning 
    Date : 2017/11/30
    代码目的：输出正常照片、复印件、手机翻拍、电脑翻拍的图片进行对比


'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from PIL import Image

plt.rcParams['font.sans-serif']=['SimHei']

width, height = 320, 240
passImg_path = r'pic/pass01.jpg'
fpImg_path = r'pic/fp05.jpg'
black_path=r'pic/black11.jpg'
fpImg_path2=r'pic/fp20.png'
#
plt.subplot(2,2,1)     #将窗口分为两行两列四个子窗口，则可显示四幅图片
passImg_ori = Image.open(passImg_path, 'r')
passImg_ori = passImg_ori.resize((width, height), Image.ANTIALIAS)
passImg_ori = np.array(passImg_ori)
title_obj = plt.title(u"(a)真实拍摄证件照")   #第一幅图片标题
# plt.getp(title_obj, 'text')            #print out the 'text' property for title
# plt.setp(title_obj)         #set the color of title to red
# plt.title(u"(a)")
# plt.xlabel('(a)')
plt.imshow(passImg_ori)      #绘制第一幅图片
plt.axis('off')

plt.subplot(2,2,2)     #第二个子窗口
passImg= Image.open(black_path, 'r')
passImg= passImg.resize((width, height), Image.ANTIALIAS)
passImg= np.array(passImg)
plt.title(u"(b)黑白复印件翻拍")   #第二幅图片标题
# plt.title(u"(b)")
# plt.xlabel('(b)')
plt.imshow(passImg)      #绘制第二幅图片,且为灰度图
plt.axis('off')     #不显示坐标尺寸

plt.subplot(2,2,3)     #将窗口分为两行两列四个子窗口，则可显示四幅图片
fpImg_ori = Image.open(fpImg_path, 'r')
fpImg_ori= fpImg_ori.resize((width, height), Image.ANTIALIAS)
fpImg_ori = np.array(fpImg_ori)
plt.title(u"(c)手机屏幕翻拍件")   #第三幅图片标题
# plt.title(u"(c)")
# plt.xlabel('(c)')
plt.imshow(fpImg_ori)      #绘制第三幅图片
plt.axis('off')

plt.subplot(2,2,4)     #第四个子窗口
fpImg= Image.open(fpImg_path2, 'r')
fpImg= fpImg.resize((width, height), Image.ANTIALIAS)
fpImg= np.array(fpImg)
plt.title(u"(d)电脑屏幕翻拍件")   #第三幅图片标题
# plt.title(u"(d)")
# plt.xlabel('(d)')
plt.imshow(fpImg)      #绘制第三幅图片
plt.axis('off')     #不显示坐标尺寸

plt.show()   #显示窗口




