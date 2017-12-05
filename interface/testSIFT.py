# coding:utf8
'''
Author : Cindy
Date : 2017/11/8
代码目的：提取sift特征

'''
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from PIL import Image
plt.rcParams['font.sans-serif']=['SimHei']

def getSift(img_path):
    '''''
    得到并查看sift特征
    '''
    # 读取图像
    img = cv2.imread(img_path)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 创建sift的类
    # sift = cv2.SIFT()
    # python3
    pointNum = 500
    siftDetector = cv2.xfeatures2d.SIFT_create(pointNum)
    # siftDetector = cv2.xfeatures2d.SURF_create(pointNum)
    # 在图像中找到关键点 也可以一步计算#kp, des = sift.detectAndCompute
    # kp = sift.detect(gray, None)
    kp, res = siftDetector.detectAndCompute(gray, None)
    # pts = []
    # for index in range(min(pointNum,50)):
        # print "kp length : {}".format(len(kp))
        # print(type(kp), type(kp[0]))
        # print kp
        # print "*" * 100
        # print res
        # Keypoint数据类型分析 http://www.cnblogs.com/cj695/p/4041399.html
        # print (kp[index].pt)
        # pts.append(kp[index].pt)
        # 计算每个点的sift
        # des = sift.compute(gray, kp)
        # print (type(kp), type(des))
        # des[0]为关键点的list，des[1]为特征向量的矩阵
        # print(type(des[0]), type(des[1]))
        # print(des[0], des[1])
        # 可以看出共有885个sift特征，每个特征为128维
        # print(des[1].shape)
        # 在灰度图中画出这些点
        # print(len(kp))
        # print(res)
    # print (np.array(pts).shape)
    # print img.shape
    img = cv2.drawKeypoints(img, outImage=img, keypoints=kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # print img.shape
    # img = cv2.drawKeypoints(gray, kp)
    # cv2.imwrite('sift_keypoints.jpg',img)
    # plt.imshow(img), plt.show()
    # 3
    cv2.imshow("sift", img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img
    # return np.array(pts)


width, height = 320, 240
passImg_path = r'pic/pass05.jpg'
fpImg_path = r'pic/fp10m.jpg'


plt.subplot(2,2,1)     #将窗口分为两行两列四个子窗口，则可显示四幅图片
passImg_ori = Image.open(passImg_path, 'r')
passImg_ori = passImg_ori.resize((width, height), Image.ANTIALIAS)
passImg_ori = np.array(passImg_ori)
plt.title(u"(a)真实拍摄证件照")   #第一幅图片标题
plt.imshow(passImg_ori)      #绘制第一幅图片
plt.axis('off')

plt.subplot(2,2,2)     #第二个子窗口
plt.title(u"(b)真实拍摄证件照SIFT特征图")   #第二幅图片标题
passImgSIFT = getSift(passImg_path)
plt.imshow(passImgSIFT)      #绘制第二幅图片,且为灰度图
plt.axis('off')     #不显示坐标尺寸

plt.subplot(2,2,3)     #将窗口分为两行两列四个子窗口，则可显示四幅图片
fpImg_ori = Image.open(fpImg_path, 'r')
fpImg_ori = np.array(fpImg_ori)
plt.title(u"(c)手机屏幕翻拍件")   #第三幅图片标题
plt.imshow(fpImg_ori)      #绘制第三幅图片
plt.axis('off')

plt.subplot(2,2,4)     #第四个子窗口
plt.title(u"(d)手机屏幕翻拍件SIFT特征图")   #第四幅图片标题
fpImgSIFT = getSift(fpImg_path)
plt.imshow(fpImgSIFT)      #绘制第四幅图片,且为灰度图
plt.axis('off')     #不显示坐标尺寸

plt.show()   #显示窗口


# getSift(passImg_path)
# getSift(fpImg_path)






# k = 10
# kmeans = KMeans(n_clusters=k)
# kmeans.fit(sift)
# print kmeans.cluster_centers_
# print kmeans.cluster_centers_[:,0]
# print kmeans.cluster_centers_[:,1]
# print kmeans.cluster_centers_.T
# print kmeans.cluster_centers_.reshape(-1).shape
# print kmeans.cluster_centers_.reshape(-1)
# plt.plot(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 'ro')
# plt.show()


# 挑选合适的K点
# K = range(1, 50)
# meandistortions = []
# for k in K:
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(sift)
#     # 求kmeans的成本函数值
#     meandistortions.append(sum(np.min(cdist(sift, kmeans.cluster_centers_, 'euclidean'), axis=1)) / sift.shape[0])
#
# plt.figure()
# plt.grid(True)
# plt1 = plt.subplot(2,1,1)
# # 画样本点
# plt1.plot(sift[:,0],sift[:,1],'k.')
# plt2 = plt.subplot(2,1,2)
# # 画成本函数值曲线
# plt2.plot(K, meandistortions, 'bx-')
# plt.show()
