# coding:utf-8
'''
    author : Cindy
    Date : 2017/11/7
    代码目的：读取数据、特征
    
'''

from sklearn.cross_validation import train_test_split
import random
from __init__ import timeDecor
import numpy as np
from features import *

passdataPath = r'../PictureData/pass_picture'
flippingPath = r'../PictureData/Flipping_pictures'
blackcopyPath = r'../PictureData/black_copyPicture'


@timeDecor
def readFeature(type='color_dis', *foldersPath):
    '''
        根据需要的类型（type）来获取对应部分的特征
    :param type:
    :param foldersPath:
    :return:
    '''
    if type == 'color_dis':
        return color_feature('dis', *foldersPath)
    elif type == 'color_histr':
        return color_feature('histr', *foldersPath)
    elif type == 'lbp':
        return LBP_feature('basic', *foldersPath)
    elif type == 'SIFT':
        return SIFT_or_SURF_feature('SIFT', *foldersPath)
    elif type == 'SURF':
        return SIFT_or_SURF_feature('SURF', *foldersPath)
    elif type == 'FFT':
        return FFT_feature(*foldersPath)
    elif type == 'all':
        return all_feature(*foldersPath)


@timeDecor
def trainValidTestSplit(imputed_X, imputed_y, itemNum=1000, times=1, train_size=0.6, valid_size=0.2, classifyNum=2):
    '''
        将带标签的数据（X,y）划分为训练、验证、测试三部分（带标签y）
    :param imputed_X:  输入的总数据X
    :param imputed_y:  输入的总数据y
    :param itemNum:   数据平衡时采样的个数
    :param times:    数据采样时训练数据与验证数据之间的比例
    :param train_size:  对总数据进行数据划分时，训练数据所占的比例
    :param valid_size:  对总数据进行数据划分时，验证数据所占的比例
    :return: 返回划分后的数据，X_bal, y_bal, X_cv, y_cv, X_test, y_test
    '''
    X, X_test, y, y_test = train_test_split(imputed_X, imputed_y, test_size=1 - (train_size + valid_size),
                                            train_size=(train_size + valid_size), random_state=22)
    X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=1 - train_size / (train_size + valid_size),
                                                    train_size=train_size / (train_size + valid_size), random_state=22)

    X_bal = []
    y_bal = []
    classifyLable = range(classifyNum)
    # for lable in classifyLable:
    #     indexName = 'index'+str(lable)
    indexs_1 = [index for index, item in enumerate(y_train) if item == 1]
    indexs_0 = [index for index in range(len(y_train)) if index not in indexs_1]
    for num in range(itemNum):
        index1 = random.choice(indexs_1)
        X_bal.append(X_train[index1])
        y_bal.append(y_train[index1])
    for num in range(int(itemNum * times)):
        index0 = random.choice(indexs_0)
        X_bal.append(X_train[index0])
        y_bal.append(y_train[index0])

    print ("X_bal shape : {}".format(np.array(X_bal).shape))
    print ("y_bal shape : {}".format(np.array(y_bal).shape))
    print ("X_cv shape : {}".format(np.array(X_cv).shape))
    print ("y_cv shape : {}".format(np.array(y_cv).shape))
    print ("X_test shape : {}".format(np.array(X_test).shape))
    print ("y_test shape : {}".format(np.array(y_test).shape))

    X_bal = np.array(X_bal)
    y_bal = np.array(y_bal)
    X_cv = np.array(X_cv)
    y_cv = np.array(y_cv)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_bal, y_bal, X_cv, y_cv, X_test, y_test
