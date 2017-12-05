# coding:utf-8
'''
    author : Cindy
    Date : 2017/11/2
    代码目的：主函数 建立模型，可选取特定的特征、分类器进行训练。其中LAB的color特征和SIFT的BOW法构成的纹理特征，拼接效果较好。
    
'''

try:
    from sklearn.model_selection import train_test_split
except:
    from sklearn.cross_validation import train_test_split
import random
from __init__ import timeDecor
import numpy as np
from features import *
from sklearn.externals import joblib
from readData import *
from classifiers import *

# 定义图片地址
passdataPath = r'../PictureData/pass_picture'
blackcopyPath = r'../PictureData/black_copyPicture2'
flippingPath = r'../PictureData/Flipping_pictures'
# 保存模型，可直接调用
GBDT_colormodelPath = r'../savedModel/GBDT_color.model'
RF_colormodelPath = r'../savedModel/RF_color.model'
GBDT_SIFTmodelPath = r'../savedModel/GBDT_SIFT.model'
RF_SIFTmodelPath = r'../savedModel/RF_SIFT.model'
GBDT_allmodelPath = r'../savedModel/GBDT_all.model'
RF_allmodelPath = r'../savedModel/RF_all.model'


@timeDecor
def main():
    # 选择需要的特征类型
    featureType = [
        # 'color_dis',  # 颜色距特征
        # 'color_histr',  # 颜色直方图特征
        # 'lbp',
        # 'SIFT',
        'SURF',
        # 'FFT',
        # 'all'  # 拼接后的总特征
    ][0]
    X, y = readFeature(
        featureType,

        passdataPath,
        blackcopyPath,
        # flippingPath,
    )
    X_bal, y_bal, X_cv, y_cv, X_test, y_test = trainValidTestSplit(X, y, itemNum=500)

    # print X_cv.shape

    gbd = GBDT_classify(X_bal, y_bal, X_cv, y_cv, X_test, y_test)
    rf = rf_classify(X_bal, y_bal, X_cv, y_cv, X_test, y_test)
    # knn = knn_classify(X_bal, y_bal, X_cv, y_cv, X_test, y_test)
    svm = svm_classify(X_bal, y_bal, X_cv, y_cv, X_test, y_test)
    # bagging_knn = bagging_knn_classify(X_bal, y_bal, X_cv, y_cv, X_test, y_test)
    # lr = lr_classify(X_bal, y_bal, X_cv, y_cv, X_test, y_test)
    # nb = nb_classify(X_bal, y_bal, X_cv, y_cv, X_test, y_test)
    # xgb = xgboost_classify(X_bal, y_bal, X_cv, y_cv, X_test, y_test)
    # da = da_classify(X_bal, y_bal, X_cv, y_cv, X_test, y_test)
    # dt = decisionTree_classify(X_bal, y_bal, X_cv, y_cv, X_test, y_test)
    # mlp = MLP_classify(X_bal, y_bal, X_cv, y_cv, X_test, y_test)
    # cnn = cnn_model(X_bal, y_bal, X_cv, y_cv, X_test, y_test)

    joblib.dump(gbd, GBDT_allmodelPath, compress=3)
    joblib.dump(rf, RF_allmodelPath, compress=3)

    # GBDT_model = joblib.load(GBDT_colormodelPath)
    # y_bal_pre = GBDT_model.predict(X_bal)
    # print y_bal_pre.shape
    # print y_bal_pre
    #
    # pre_y_bal = GBDT_model.predict(X_bal)
    # pre_y_cv = GBDT_model.predict(X_cv)
    # pre_y_test = GBDT_model.predict(X_test)
    # print("GBDT_classify    train Metrics : {0}".format(PRF(y_bal, pre_y_bal)))
    # print("GBDT_classify    cv Metrics : {0}".format(PRF(y_cv, pre_y_cv)))
    # print("GBDT_classify    test Metrics : {0}".format(PRF(y_test, pre_y_test)))
    # print("Test PRF : {0}".format(precision_recall_fscore_support(y_test, pre_y_test)))
    # print('The Accuracy of ' + 'GBDT' + ' is :', GBDT_model.score(X_test, y_test))
    # print(classification_report(y_test, pre_y_test))


if __name__ == "__main__":
    main()
