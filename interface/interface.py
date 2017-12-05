# coding:utf-8
'''
    author : minning 
    Date : 2017/11/21
    代码目的：
    
'''
# from ..model import LBP
# from ..model import features,classifiers
# from ..model.features import color_moments

from PIL import Image
import cv2
import numpy as np
from sklearn.externals import joblib
import pickle
from scipy.cluster.vq import kmeans, vq


def getResult(picPath, type='all'):

    pic = process_size(picPath)
    color_feat = get_color(np.array(pic))
    color_feat = np.array(color_feat)
    color_feat = np.reshape(color_feat,(1,9))
    SIFT_feat = get_SIFT(picPath)
    all_feature = np.concatenate((color_feat, SIFT_feat), axis=1)
    print "color_feat shape ； {}".format(color_feat.shape)
    print "SIFT_feat shape ； {}".format(SIFT_feat.shape)
    print "all_feature shape ； {}".format(all_feature.shape)
    GBDT_color_modelPath = r'../savedModel/GBDT_color.model'
    GBDT_SIFT_modelPath = r'../savedModel/GBDT_SIFT.model'
    GBDT_all_modelPath = r'../savedModel/GBDT_all.model'
    if type=='color':
        clf = joblib.load(GBDT_color_modelPath)
        result = clf.predict(color_feat)
    elif type=='sift':
        clf = joblib.load(GBDT_SIFT_modelPath)
        result = clf.predict(SIFT_feat)
    elif type=='all':
        clf = joblib.load(GBDT_all_modelPath)
        print all_feature.shape
        result = clf.predict(all_feature)

    if result[0]==0:
        ret = "此照片合格"
    elif result[0]==1:
        ret = "此照片不合格，是复印件"
    elif result[0]==2:
        ret = "此照片不合格，是手机或电脑屏幕复印件"

    return ret

def process_size(picPath):
    img = Image.open(picPath,'r')  # 读取和代码处于同一目录下的 lena.png
    width, height = 214, 135  # 用PIL进行尺寸修改,因为后面要计算总的 像素值，所以这里设置的小一点。
    out = img.resize((width, height), Image.ANTIALIAS)  # resize image with high-quality

    return out


def get_color(pic, colorType='lab'):
    img = pic
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert BGR to LAB colorspace
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # Convert BGR to HSV colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if colorType == 'rgb':
        myColorType = rgb
    elif colorType == 'lab':
        myColorType = lab
    elif colorType == 'hsv':
        myColorType = hsv
    # Split the channels - h,s,v
    h, s, v = cv2.split(myColorType)
    # Initialize the color feature
    color_feature = []
    # N = h.shape[0] * h.shape[1]
    # The first central moment - average
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])
    # The second central moment - standard deviation
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.extend([h_std, s_std, v_std])
    # The third central moment - the third root of the skewness
    h_skewness = np.mean(abs(h - h.mean()) ** 3)
    s_skewness = np.mean(abs(s - s.mean()) ** 3)
    v_skewness = np.mean(abs(v - v.mean()) ** 3)
    h_thirdMoment = h_skewness ** (1. / 3)
    s_thirdMoment = s_skewness ** (1. / 3)
    v_thirdMoment = v_skewness ** (1. / 3)
    color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])

    return color_feature


def get_SIFT(picPath):
    img = Image.open(picPath, 'r')
    img = np.array(img)
    pointNum = 300
    sift = cv2.xfeatures2d.SIFT_create(pointNum)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, des = sift.detectAndCompute(gray, None)
    cluster_nums = 1024
    im_features = np.zeros((1, cluster_nums), "float32")
    file_path = 'voc'
    voc = pickle.load(open(file_path))
    words, distance = vq(des, voc)
    for w in words:
        im_features[0][w] += 1

    print "im_features shape : {}".format(im_features.shape)  # (1L, 1024L)

    return im_features


def main():
    # picPath = '../PictureData/black_copyPicture/13.png'
    # picPath = '../PictureData/pass_picture/309.jpg'
    # picPath = r'D:\Tencent\fileReceive\11.20带回\11.20\翻拍\getImgPat8h (14).png'
    picPath = r'pic/pass04.jpg'
    # result = get_SIFT(picPath)
    result = getResult(picPath, type='color')
    print result


if __name__ == "__main__":
    main()
