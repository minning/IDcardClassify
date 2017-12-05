# coding:utf-8
'''
    author : minning 
    Date : 2017/10/21
    代码目的：提取图片的FFT特征
    
'''



import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVR
from skimage import feature as skft
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from getData import getData2
import cv2
import os
import warnings
warnings.filterwarnings("ignore")


radius = 1
n_point = radius * 8
# 训练数据地址
passdataPath = '../PictureData/pass_picture'
flippingPath = '../PictureData/Flipping_pictures'
train_blackcopy = '../PictureData/black_copyPicture'

# 测试数据地址
testpassPath = '../testdata/testpass'
testflipdataPath = '../testdata/testflipPicture'
test_blackcopyPicture = '../testdata/testblackcopy'
# 全是翻拍的：1


# LBP方法提取图片的纹理特征
def FFT_feature(passFolderPath, filterFolder_path):

    X, y = [], []

    # 训练数据
    for file1 in os.listdir(passFolderPath):
        fileAbsPath1 = os.path.join(passFolderPath, file1)
        img = cv2.imread(fileAbsPath1, 1)
        img_R, img_G, img_B = cv2.split(img)
        f_R = np.fft.fft2(img_R)
        f_G = np.fft.fft2(img_G)
        f_B = np.fft.fft2(img_B)
        # f = np.fft.fft2(img)
        f_RGB = np.concatenate((f_R,f_G,f_B),axis=1)

        X.append(f_RGB)
        y.append(0)

    # 测试数据
    for file2 in os.listdir(filterFolder_path):
        fileAbsPath2 = os.path.join(filterFolder_path, file2)
        img = cv2.imread(fileAbsPath2, 1)
        img_R, img_G, img_B = cv2.split(img)
        f_R = np.fft.fft2(img_R)
        f_G = np.fft.fft2(img_G)
        f_B = np.fft.fft2(img_B)
        f_RGB = np.concatenate((f_R, f_G, f_B), axis=1)
        X.append(f_RGB)
        y.append(1)

    X = np.array(X)
    y = np.array(y)

    print "X Shape : {}".format(X.shape)
    print "y Shape : {}".format(y.shape)

    return np.array(X), np.array(y)





if __name__ == "__main__":
    x_train, y_train = FFT_feature(passdataPath, flippingPath)
    x_test, y_test = FFT_feature(testpassPath, testflipdataPath)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
    print ("x_train shape : {}".format(x_train.shape))


    rfc = RandomForestClassifier(n_estimators=300)
    rfc.fit(x_train, y_train)
    train_predict = rfc.predict(x_train)

    print ("y_train")
    print (y_train)
    print ('train_predict')
    print (train_predict)
    print ("Train PRF : {0}".format(precision_recall_fscore_support(y_train, train_predict)))
    # test_predict = svr_rbf.predict(x_test)
    test_predict = rfc.predict(x_test)
    # test_predict = gbc.predict(x_test)
    print ("y_test")
    print (y_test)
    print ('test_predict')
    print (test_predict)
    print ("Test PRF : {0}".format(precision_recall_fscore_support(y_test, test_predict)))
    # print ('The Accuracy of SVC is :',svr_rbf.score(x_test,y_test))
    print ('The Accuracy of RF is :',rfc.score(x_test,y_test))
    # print ('The Accuracy of SVC is :',gbc.score(x_test,y_test))
    print (classification_report(y_test,test_predict))
