# coding:utf-8
'''
    author : Cindy
    Date : 2017/11/7
    代码目的：模型的特征提取，其中包括颜色、纹理、频谱特征
    
'''
import os
import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq
from sklearn import preprocessing
from LBP import LBP
import pickle


# 颜色距的方法,颜色特征的提取
# Compute low order moments(1,2,3)
def color_dis_moments(filename, colorType='lab'):
    '''
        颜色特征主要包括三类：RGB、HSV、LAB特征,选择相应的colorType即可对相应颜色特征。
        目前效果最好的是lab类型的特征在RF上的效果，F值为（0.99, 0.86）
        其次hsv类型的color特征也还不错，在KNN上的F值为（0.97, 0.73）
        RGB特征的效果就是原始图像的效果，效果较差
    :param filename:
    :return:
    '''
    img = cv2.imread(filename)
    # print "img"
    # print img
    if img is None:
        return
    # img = cv2.blur(img, (3, 3))
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

    color_feature = np.array(color_feature)

    return color_feature


def color_histr_moments(filename, colorType='lab'):
    '''
        颜色特征主要包括三类：RGB、HSV、LAB特征,选择相应的colorType即可对相应颜色特征。
        目前效果最好的是lab类型的特征在RF上的效果，F值为（0.99, 0.86）
        其次hsv类型的color特征也还不错，在KNN上的F值为（0.97, 0.73）
        RGB特征的效果就是原始图像的效果，效果较差
    :param filename:
    :return:
    '''
    if colorType == 'rgb':
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
    elif colorType == 'hsv':
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif colorType == 'lab':
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    histr_0 = cv2.calcHist([img], [0], None, [256], [0, 256])
    histr_1 = cv2.calcHist([img], [1], None, [256], [0, 256])
    histr_2 = cv2.calcHist([img], [2], None, [256], [0, 256])
    histr = np.concatenate((histr_0, histr_1, histr_2), axis=0)
    histr = np.reshape(histr, (histr.shape[0]))

    return histr


def color_feature(type, *foldersPath):
    '''
    获取指定多個路径下的图片的颜色特征
    :param *folderPath: 可变参数类型的多个文件夹路径，其中每一个文件夹中包含同一个类别的图片数据
    :return: X，y分别代表
    '''
    X, y = [], []
    lable = 0
    for oneFolderPath in foldersPath:
        # print str(lable)+oneFolderPath
        for file in os.listdir(oneFolderPath):
            fileAbsPath = os.path.join(oneFolderPath, file)
            if type == 'dis':
                oneColorFeat = color_dis_moments(fileAbsPath)
            elif type == 'histr':
                oneColorFeat = color_histr_moments(fileAbsPath)
            X.append(oneColorFeat)
            y.append(lable)
        lable += 1

    return np.array(X), np.array(y)


# 获取图片LBP纹理特征
def getLBP_feature(filename, lbpType='basic'):
    lbp = LBP()
    img = cv2.imread(filename)
    if img is None:
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if lbpType == 'basic':
        lbpFeature = lbp.lbp_basic(gray)
    elif lbpType == 'revolve':
        lbpFeature = lbp.lbp_revolve(gray)
    elif lbpType == 'uniform':
        lbpFeature = lbp.lbp_uniform(gray)
    else:
        # lbpType == 'revolve_uniform':
        lbpFeature = lbp.lbp_revolve_uniform(gray)

    # print ("lbpFeature shape : {}".format(lbpFeature.shape))
    return lbpFeature


def LBP_feature(lbpType='basic', *foldersPath):
    '''
    获取指定多個路径下的图片的颜色特征
    :param *folderPath: 可变参数类型的多个文件夹路径，其中每一个文件夹中包含同一个类别的图片数据
    :return: X，y分别代表
    '''
    X, y = [], []
    lable = 0
    for oneFolderPath in foldersPath:
        for file in os.listdir(oneFolderPath):
            fileAbsPath = os.path.join(oneFolderPath, file)
            oneLbpFeat = getLBP_feature(fileAbsPath, lbpType=lbpType)
            X.append(oneLbpFeat)
            y.append(lable)
        lable += 1

    return np.array(X), np.array(y)


def getFFT_feature(filename):
    '''
        获取单个图片的FFT特征
    :param filename:
    :return:
    '''
    from PIL import Image
    img = cv2.imread(filename)
    width, height = 160, 120  # 用PIL进行尺寸修改,因为后面要计算总的 像素值，所以这里设置的小一点。
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    # img = img.resize((width, height), Image.ANTIALIAS)  # resize image with high-quality
    img_R, img_G, img_B = cv2.split(img)

    # f_R = np.fft.fft2(img_R)
    # f_R = np.fft.fftshift(f_R)  # 得到结果为复数
    # f_R = 20 * np.log(np.abs(f_R))  # 先取绝对值，表示取模。取对数，将数据范围变小
    # f_G = np.fft.fft2(img_G)
    # f_G = np.fft.fftshift(f_G)  # 得到结果为复数
    # f_G = 20 * np.log(np.abs(f_G))  # 先取绝对值，表示取模。取对数，将数据范围变小
    # f_B = np.fft.fft2(img_B)
    # f_B = np.fft.fftshift(f_B)  # 得到结果为复数
    # f_B = 20 * np.log(np.abs(f_B))  # 先取绝对值，表示取模。取对数，将数据范围变小

    f_R = abs(np.fft.fft2(img_R))
    f_G = abs(np.fft.fft2(img_G))
    f_B = abs(np.fft.fft2(img_B))
    f_RGB = np.concatenate((f_R, f_G, f_B), axis=1)
    # print "f_RGB shape : {}".format(f_RGB.shape)
    f_RGB = f_RGB.T.reshape(-1)

    return f_RGB


def FFT_feature(*foldersPath):
    X, y = [], []
    lable = 0
    for oneFolderPath in foldersPath:
        for file in os.listdir(oneFolderPath):
            fileAbsPath = os.path.join(oneFolderPath, file)
            oneLbpFeat = getFFT_feature(fileAbsPath)
            X.append(oneLbpFeat)
            y.append(lable)
        lable += 1
    # print "Before"
    # print "X shape : {}".format(np.array(X).shape)
    X = np.array(X)
    # X = np.reshape(X, (X.shape[0], 1, X.shape[1], X.shape[2]))
    # print "After"
    # print "X shape : {}".format(X.shape)



    return np.array(X), np.array(y)


def getOldSIFT_feature(filename):
    '''''
        得到并查看sift特征
        '''
    # 读取图像
    img = cv2.imread(filename)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 创建sift的类
    # sift = cv2.SIFT()
    # python3
    pointNum = 300
    siftDetector = cv2.xfeatures2d.SIFT_create(pointNum)
    # 在图像中找到关键点 也可以一步计算#kp, des = sift.detectAndCompute
    # kp = sift.detect(gray, None)
    kp, res = siftDetector.detectAndCompute(gray, None)
    pts = []
    for index in range(min(pointNum, 30)):
        # print "kp length : {}".format(len(kp))
        # print(type(kp), type(kp[0]))
        # print kp
        # print "*" * 100
        # print res
        # Keypoint数据类型分析 http://www.cnblogs.com/cj695/p/4041399.html
        # print index
        # print (kp[index].pt)
        pts.append(kp[index].pt)
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
    # print np.array(pts).shape

    # print img.shape
    # img = cv2.drawKeypoints(img, outImage=img, keypoints=kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # print img.shape
    # img = cv2.drawKeypoints(gray, kp)
    # cv2.imwrite('sift_keypoints.jpg',img)
    # plt.imshow(img), plt.show()
    # 3
    # cv2.imshow("sift", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return np.array(pts)


def get_file_name(path):
    '''
    Args: path to list;  Returns: path with filenames
    '''
    filenames = os.listdir(path)
    path_filenames = []
    filename_list = []
    for file in filenames:
        if not file.startswith('.'):
            path_filenames.append(os.path.join(path, file))
            filename_list.append(file)

    return path_filenames


# 得到图片SIFT表述的特征
def getSIFT_or_SURF_feature(paths, type='SIFT', cluster_nums=1024, pointNum=300, randomState=None):
    '''
    :param file_list:
    :param cluster_nums:
    :param randomState:
    :return:
    '''
    file_list = get_file_name(paths)
    files = file_list
    # sift = cv2.SIFT()
    if type == 'SIFT':
        sift = cv2.xfeatures2d.SIFT_create(pointNum)
    elif type == 'SURF':
        sift = cv2.xfeatures2d.SURF_create(pointNum)

    des_list = []
    for file in files:
        img = cv2.imread(file)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, des = sift.detectAndCompute(gray, None)
        # print "des shape : {}".format(des.shape)
        if des is None:
            file_list.remove(file)
            continue
        des_list.append((file, des))

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))
    print "descriptors shape : {}".format(descriptors.shape)
    voc, variance = kmeans(descriptors, cluster_nums, 1)
    print "voc shape : {}".format(voc.shape)
    # print voc
    print "variance shape : {}".format(variance.shape)

    voc_path = 'voc'
    pickle.dump(voc, open(voc_path, 'w'))
    # 通过dump把处理好的数据序列化

    im_features = np.zeros((len(files), cluster_nums), "float32")
    for i in range(len(des_list)):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            im_features[i][w] += 1

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
    idf = np.array(np.log((1.0 * len(files) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

    # Perform L2 normalization
    im_features = im_features * idf
    im_features = preprocessing.normalize(im_features, norm='l1')

    print ("im_features shape : {}".format(im_features.shape))

    return im_features


def SIFT_or_SURF_feature(type='SIFT', *foldersPath):
    '''
        获取指定多個路径下的图片的SIFT特征
        :param *folderPath: 可变参数类型的多个文件夹路径，其中每一个文件夹中包含同一个类别的图片数据
        :return: X，y分别代表
        '''
    X, y = [], []
    flag = False
    lable = 0
    for oneFolderPath in foldersPath:
        # 获取一个文件夹下所有图片的SIFT特征矩阵
        feat = getSIFT_or_SURF_feature(oneFolderPath, type=type, cluster_nums=1024, pointNum=400)
        if not flag:
            X = feat
            flag = True
        else:
            X = np.concatenate((X, feat), axis=0)
        for i in range(feat.shape[0]):
            y.append(lable)
        lable += 1

    return np.array(X), np.array(y)


# def FFT_feature(*foldersPath):
#     '''
#         获取指定多個路径下的图片的SIFT特征
#         :param *folderPath: 可变参数类型的多个文件夹路径，其中每一个文件夹中包含同一个类别的图片数据
#         :return: X，y分别代表
#         '''
#     X, y = [], []
#     flag = False
#     lable = 0
#     for oneFolderPath in foldersPath:
#         # SIFT特征
#         feat = getFFT_feature(oneFolderPath, 1024)
#         if not flag:
#             X = feat
#             flag = True
#         else:
#             X = np.concatenate((X, feat),axis=0)
#         for i in range(feat.shape[0]):
#             y.append(lable)
#         lable += 1
#
#     return np.array(X), np.array(y)


# 特征拼接
def all_feature(*foldersPath):
    '''
    获取指定多個路径下的图片的颜色特征
    :param *folderPath: 可变参数类型的多个文件夹路径，其中每一个文件夹中包含同一个类别的图片数据
    :return: X，y分别代表
    '''
    x_color_feat, y = color_feature(*foldersPath)
    # x_LBP_feat, _ = LBP_feature('basic', *foldersPath)
    x_SIFT_feat, _ = SIFT_or_SURF_feature('SIFT', *foldersPath)
    x_SURF_feat, _ = SIFT_or_SURF_feature('SURF', *foldersPath)
    X = np.concatenate((x_color_feat, x_SURF_feat), axis=1)

    print ("x_color_feat shape : {}".format(x_color_feat.shape))
    print ("x_SIFT_feat shape : {}".format(x_SIFT_feat.shape))
    print ("x_SURF_feat shape : {}".format(x_SURF_feat.shape))
    print ("X shape : {}".format(X.shape))

    return np.array(X), np.array(y)
