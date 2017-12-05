# coding:utf-8
'''
    author : minning
    Date : 2017/11/7
    代码目的：采用的12种分类器，同时进行，选取最佳效果，评判标准是PRF（精确率precision，召回率recall，F值，数值都是越大越好）。
    
'''


from sklearn.ensemble import RandomForestClassifier
from __init__ import timeDecor
from sklearn.metrics import precision_recall_fscore_support as PRF
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Dropout, Activation, Lambda, merge, Flatten
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers import TimeDistributed
from keras.layers import GlobalMaxPooling1D, Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn import metrics


@timeDecor
#随机森林
def rf_classify(X_train, y_train, X_cv, y_cv, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(random_state=0, n_estimators=100)

    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    pre_y_cv = clf.predict(X_cv)
    pre_y_test = clf.predict(X_test)
    print("rf train Metrics : {0}".format(PRF(y_train, pre_y_train)))
    print("rf cv Metrics : {0}".format(PRF(y_cv, pre_y_cv)))
    print("rf test Metrics : {0}".format(PRF(y_test, pre_y_test)))
    print("Test PRF : {0}".format(precision_recall_fscore_support(y_test, pre_y_test)))
    print('The Accuracy of ' + 'rf' + ' is :', clf.score(X_test, y_test))
    print(classification_report(y_test, pre_y_test))
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, pre_y_test)
    # print metrics.auc(fpr, tpr)

    return clf


@timeDecor
#SVM
def svm_classify(X_train, y_train, X_cv, y_cv, X_test, y_test):
    from sklearn import svm

    clf = svm.SVC(kernel='linear', C=1e4, gamma=0.01)
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    pre_y_cv = clf.predict(X_cv)
    pre_y_test = clf.predict(X_test)
    print("svm train Metrics : {0}".format(PRF(y_train, pre_y_train)))
    print("svm cv Metrics : {0}".format(PRF(y_cv, pre_y_cv)))
    print("svm test Metrics : {0}".format(PRF(y_test, pre_y_test)))
    print("Test PRF : {0}".format(precision_recall_fscore_support(y_test, pre_y_test)))
    print('The Accuracy of ' + 'SVC' + ' is :', clf.score(X_test, y_test))
    print(classification_report(y_test, pre_y_test))

    return clf


@timeDecor
#knn
def knn_classify(X_train, y_train, X_cv, y_cv, X_test, y_test):
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(n_neighbors=5)

    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    pre_y_cv = clf.predict(X_cv)
    pre_y_test = clf.predict(X_test)
    print("knn train Metrics : {0}".format(PRF(y_train, pre_y_train)))
    print("knn cv Metrics : {0}".format(PRF(y_cv, pre_y_cv)))
    print("knn test Metrics : {0}".format(PRF(y_test, pre_y_test)))
    print("Test PRF : {0}".format(precision_recall_fscore_support(y_test, pre_y_test)))
    print('The Accuracy of ' + 'KNN' + ' is :', clf.score(X_test, y_test))
    print(classification_report(y_test, pre_y_test))

    return clf


@timeDecor
#Bagging meta-estimator
def bagging_knn_classify(X_train, y_train, X_cv, y_cv, X_test, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import BaggingClassifier

    clf = BaggingClassifier(KNeighborsClassifier(),
                            max_samples=0.5, max_features=0.5)
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    pre_y_cv = clf.predict(X_cv)
    pre_y_test = clf.predict(X_test)
    print("bagging_knn  train Metrics : {0}".format(PRF(y_train, pre_y_train)))
    print("bagging_knn  cv Metrics : {0}".format(PRF(y_cv, pre_y_cv)))
    print("bagging_knn  test Metrics : {0}".format(PRF(y_test, pre_y_test)))
    print("Test PRF : {0}".format(precision_recall_fscore_support(y_test, pre_y_test)))
    print('The Accuracy of ' + 'bagging_knn' + ' is :', clf.score(X_test, y_test))
    print(classification_report(y_test, pre_y_test))

    return clf


@timeDecor
#逻辑回归
def lr_classify(X_train, y_train, X_cv, y_cv, X_test, y_test):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=1e5)
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    pre_y_cv = clf.predict(X_cv)
    pre_y_test = clf.predict(X_test)
    print("lr train Metrics : {0}".format(PRF(y_train, pre_y_train)))
    print("lr cv Metrics : {0}".format(PRF(y_cv, pre_y_cv)))
    print("lr test Metrics : {0}".format(PRF(y_test, pre_y_test)))
    print("Test PRF : {0}".format(precision_recall_fscore_support(y_test, pre_y_test)))
    print('The Accuracy of ' + 'bagging_knn' + ' is :', clf.score(X_test, y_test))
    print(classification_report(y_test, pre_y_test))

    return clf


@timeDecor
#贝叶斯
def nb_classify(X_train, y_train, X_cv, y_cv, X_test, y_test):
    from sklearn.naive_bayes import GaussianNB

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    pre_y_cv = clf.predict(X_cv)
    pre_y_test = clf.predict(X_test)
    print("nb  train Metrics : {0}".format(PRF(y_train, pre_y_train)))
    print("nb  cv Metrics : {0}".format(PRF(y_cv, pre_y_cv)))
    print("nb  test Metrics : {0}".format(PRF(y_test, pre_y_test)))
    print("Test PRF : {0}".format(precision_recall_fscore_support(y_test, pre_y_test)))
    print('The Accuracy of ' + 'nb' + ' is :', clf.score(X_test, y_test))
    print(classification_report(y_test, pre_y_test))

    return clf


@timeDecor
#二次判别分析
def da_classify(X_train, y_train, X_cv, y_cv, X_test, y_test):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    pre_y_cv = clf.predict(X_cv)
    pre_y_test = clf.predict(X_test)
    print("da   train Metrics : {0}".format(PRF(y_train, pre_y_train)))
    print("da   cv Metrics : {0}".format(PRF(y_cv, pre_y_cv)))
    print("da   test Metrics : {0}".format(PRF(y_test, pre_y_test)))
    print("Test PRF : {0}".format(precision_recall_fscore_support(y_test, pre_y_test)))
    print('The Accuracy of ' + 'da' + ' is :', clf.score(X_test, y_test))
    print(classification_report(y_test, pre_y_test))

    return clf


@timeDecor
#决策树
def decisionTree_classify(X_train, y_train, X_cv, y_cv, X_test, y_test):
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    pre_y_cv = clf.predict(X_cv)
    pre_y_test = clf.predict(X_test)
    print("DT  train Metrics : {0}".format(PRF(y_train, pre_y_train)))
    print("DT  cv Metrics : {0}".format(PRF(y_cv, pre_y_cv)))
    print("DT  test Metrics : {0}".format(PRF(y_test, pre_y_test)))
    print("Test PRF : {0}".format(precision_recall_fscore_support(y_test, pre_y_test)))
    print('The Accuracy of ' + 'decisionTree' + ' is :', clf.score(X_test, y_test))
    print(classification_report(y_test, pre_y_test))

    return clf


@timeDecor
#大规模并行boosted tree
def xgboost_classify(X_train, y_train, X_cv, y_cv, X_test, y_test):
    import xgboost

    clf = xgboost.XGBClassifier()
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    pre_y_cv = clf.predict(X_cv)
    pre_y_test = clf.predict(X_test)
    print("xgboost    train Metrics : {0}".format(PRF(y_train, pre_y_train)))
    print("xgboost    cv Metrics : {0}".format(PRF(y_cv, pre_y_cv)))
    print("xgboost    test Metrics : {0}".format(PRF(y_test, pre_y_test)))
    print("Test PRF : {0}".format(precision_recall_fscore_support(y_test, pre_y_test)))
    print('The Accuracy of ' + 'xgboost' + ' is :', clf.score(X_test, y_test))
    print(classification_report(y_test, pre_y_test))

    return clf


@timeDecor
#梯度提升树
def GBDT_classify(X_train, y_train, X_cv, y_cv, X_test, y_test):
    from sklearn.ensemble import GradientBoostingClassifier

    clf = GradientBoostingClassifier(n_estimators=200)
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    pre_y_cv = clf.predict(X_cv)
    pre_y_test = clf.predict(X_test)
    print("GBDT_classify    train Metrics : {0}".format(PRF(y_train, pre_y_train)))
    print("GBDT_classify    cv Metrics : {0}".format(PRF(y_cv, pre_y_cv)))
    print("GBDT_classify    test Metrics : {0}".format(PRF(y_test, pre_y_test)))
    print("Test PRF : {0}".format(precision_recall_fscore_support(y_test, pre_y_test)))
    print('The Accuracy of ' + 'GBDT' + ' is :', clf.score(X_test, y_test))
    print(classification_report(y_test, pre_y_test))

    return clf


@timeDecor
#模型聚合-投票
def voting_classify(X_train, y_train, X_cv, y_cv, X_test, y_test):
    from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB

    clf1 = GradientBoostingClassifier(n_estimators=200)
    clf2 = RandomForestClassifier(random_state=0, n_estimators=500)
    clf3 = LogisticRegression(random_state=1)
    clf4 = GaussianNB()
    clf = VotingClassifier(estimators=[
        ('gbdt',clf1),
        ('rf', clf2),
        ('lr',clf3),
        ('nb',clf4),
        # ('xgboost',clf5),
    ],
        voting='soft'
    )
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    pre_y_cv = clf.predict(X_cv)
    pre_y_test = clf.predict(X_test)
    print("voting_classify    train Metrics : {0}".format(PRF(y_train, pre_y_train)))
    print("voting_classify    cv Metrics : {0}".format(PRF(y_cv, pre_y_cv)))
    print("voting_classify    test Metrics : {0}".format(PRF(y_test, pre_y_test)))
    print("Test PRF : {0}".format(precision_recall_fscore_support(y_test, pre_y_test)))
    print('The Accuracy of ' + 'Voting' + ' is :', clf.score(X_test, y_test))
    print(classification_report(y_test, pre_y_test))

    return clf


@timeDecor
#多层感知机
def MLP_classify(X_train, y_train, X_cv, y_cv, X_test, y_test):
    from sklearn.neural_network import MLPClassifier
    from sklearn.neural_network import MLPClassifier
    t0 = time()
    clf = MLPClassifier(alpha=1)
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    pre_y_cv = clf.predict(X_cv)
    pre_y_test = clf.predict(X_test)
    print("voting_classify    train Metrics : {0}".format(PRF(y_train, pre_y_train)))
    print("voting_classify    cv Metrics : {0}".format(PRF(y_cv, pre_y_cv)))
    print("voting_classify    test Metrics : {0}".format(PRF(y_test, pre_y_test)))
    print("Test PRF : {0}".format(precision_recall_fscore_support(y_test, pre_y_test)))
    print('The Accuracy of ' + 'MLP' + ' is :', clf.score(X_test, y_test))
    print(classification_report(y_test, pre_y_test))

    return clf


@timeDecor
#卷积神经网络
def cnn_model(X_train, y_train, X_cv, y_cv, X_test, y_test):
    '''
    Building this model with CNN
    '''
    # 所有数据的输入形状
    max_npara = X_train.shape[2]
    max_nsent = X_train.shape[3]
    para_conv = 3
    sent_conv = 3
    embedding_dim = 100
    maxlen = 400
    # number of convolutional filters to use
    nb_filters = 32
    # convolution kernel size
    nb_conv = 3
    # size of pooling area for max pooling
    nb_pool = 2
    batch_size = 128
    nb_classes = 10
    nb_epoch = 10
    iteraStartEpoch = 50
    iterEndEpoch = 81
    n_class = 4

    # 段落模型
    essay_input = Input(shape=(max_npara, max_nsent, embedding_dim))
    max_features = 20000
    embedding_layer = Embedding(max_features, 128, input_length=maxlen)
    x = Convolution2D(nb_filters, para_conv, sent_conv, name='conv1',
                      border_mode='valid')(embedding_layer)
    x = Activation('relu')(x)
    x = Convolution2D(nb_filters, para_conv, sent_conv, name='conv2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(nb_pool, nb_pool))(x)
    # x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(n_class)(x)
    prediction = Activation('softmax')(x)

    essay_model = Model(input=essay_input, output=prediction)
    essay_model.compile(loss='categorical_crossentropy',
                        optimizer='adadelta',
                        metrics=['accuracy'])
    # print(essay_model.summary())
    # plot(essay_model, to_file='cnn_model.png')

    essay_model.fit(X_train, y_train, batch_size=16, epochs=nb_epoch,
              validation_data=(X_test, y_test))
    pre_y_train = essay_model.predict(X_train)
    pre_y_cv = essay_model.predict(X_cv)
    pre_y_test = essay_model.predict(X_test)
    print("voting_classify    train Metrics : {0}".format(PRF(y_train, pre_y_train)))
    print("voting_classify    cv Metrics : {0}".format(PRF(y_cv, pre_y_cv)))
    print("voting_classify    test Metrics : {0}".format(PRF(y_test, pre_y_test)))


    return essay_model