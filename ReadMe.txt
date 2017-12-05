环境：Python 3.5
sklearn
numpy
cv2
PIL
此程序针对通过图片、翻拍图片、复印件进行识别分类。
主要运行函数:运行process_size.py对图片进行Resize和重命名后，再运行main.py
1、所有的图片都放在PictureData文件夹中，分为三部分：通过图片pass_picture（598张）；翻拍图片Flipping_pictures（191张），黑白复印件图片black_copyPicture（45张）
   preprocess文件夹中process_size.py：对图片进行预处理：Resize和重命名。
2、数据集编列索引：图像描述符（image descriptor，也称描述子）提取每幅图像的特征。这里采用的是颜色特征、纹理特征、频谱特征。
   model文件夹中的features.py:包括对图片颜色特征（LAB,HSV,RGB）、纹理特征（SIFT,LBP)、频谱特征（FFT）的提取，其中两个或三个特征可以任意拼接，作为训练集的特征数据。
   其中颜色特征（LAB颜色空间的颜色距）对于通过图片和复印件图片的分类效果较好，0.99；BOW方法的SIFT特征对通过图片和翻拍图片的分类效果较好0.98
   将两个特征拼接，进行通过图片、翻拍图片、复印图片，分别标注为0,1,2，三分类识别。
3、基于标签和所述图片特征构建训练集，采用分类器模型在所述训练集上训练出分类模型。
   classifiers.py：分类器便于比较得到最适合的分类器。

