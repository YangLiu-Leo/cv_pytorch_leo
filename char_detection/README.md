# cv_pytorch_learn

### 一　python虚拟环境使用

#### 1.1 虚拟环境

本地创建和使用一个虚拟环境，pip list可以看到，系统里python下面的很多包在虚拟环境里都没有

```
python3.7 -m venv py37_cv　新建
source py37_cv/bin/activate　激活
deactivate　退出
pip freeze > requirements.txt　打包新安装的包版本
pip install -r requirements.txt　在另一台机器上新建虚拟环境，复制上一个环境
```



#### 1.2 Docker



### 二　天池比赛  不定长街景字符编码识别（代码在char_detection文件夹里）

**赛题目的：**识别图片中的街道字符编码

**赛题数据集：**数据采用公开数据集SVHN街道字符，并进行了匿名采样处理，训练集数据包括3W张照片，验证集数据包括1W张照片。

**数据集标签：**需要注意的是本赛题需要选手识别图片中所有的字符，为了降低比赛难度，提供了训练集、验证集和测试集中所有字符的位置框，数据集标签采用json文件保存。对于训练数据每张图片将给出对于的编码标签，和具体的字符框的位置（训练集、测试集和验证集都给出字符位置），可用于模型训练：

| Field  | Description                |      |
| ------ | -------------------------- | ---- |
| top    | 左上角坐标X                |      |
| height | 字符高度                   |      |
| left   | 左上角坐标Y                |      |
| width  | 字符宽度                   |      |
| label  | 字符编码（每个数字的真值） |      |



学员手册：https://shimo.im/docs/ne3VVNlN1Js8FB3b/read

课程内容、任务及baseline：　https://github.com/datawhalechina/team-learning/tree/master/03%20%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%



#### Task1 赛题思路和数据集熟悉

#### 1.1 虚拟环境及依赖包的安装，及baseline的运行步骤

１）可以先创建虚拟环境，然后执行下面指令复现环境

pip install -r requirements.txt

２）download_dataset.py　可以从云端下次数据集

３）json_load.py　可以读一张图片然后根据json文件里的label把数字扣出来

４）baseline.py　可以训练模型，然后输出识别结果到.csv文件(可以WPS打开)里

#### 1.2 pytorch源码学习

Pytorch Sampler详解(界面动画很赞)：　https://www.cnblogs.com/marsggbo/p/11541054.html  

DataLoader源码阅读：　https://blog.csdn.net/u012436149/article/details/78545766 

#### 1.3 解题思路

赛题本质是分类问题，需要对图片的字符进行识别。本次赛题的难点是需要对不定长的字符进行识别，与传统的图像分类任务有所不同。

**简单入门思路：**定长字符识别，填充字符长度到定长

**专业字符识别思路：**不定长字符识别，比较典型的有CRNN字符识别模型，可以视为单词或句子

**专业分类思路：**检测再识别  

在赛题数据中已经给出了训练集、验证集中所有图片中字符的位置，因此可以首先将字符的位置进行识别，利用物体检测的思路完成。此种思路需要参赛选手构建字符检测模型，对测试集中的字符进行识别。选手可以参考物体检测模型SSD或者YOLO来完成。 



#### Task2　图像读取和数据扩增（Data Augmentation）的理解

#### 2.1 图像读取

由于赛题数据是图像数据，赛题的任务是识别图像中的字符。因此我们首先需要完成对数据的读取操作，在Python中有很多库可以完成数据读取的操作，比较常见的有Pillow(RGB)和OpenCV(BGR格式，貌似需要转成RGB才能显示).

Pillow有很多图像操作，是图像处理的必备库。       
Pillow的官方文档：https://pillow.readthedocs.io/en/stable/

OpenCV包含了众多的图像处理的功能，OpenCV包含了你能想得到的只要与图像相关的操作。此外OpenCV还内置了很多的图像特征处理算法，如关键点检测、边缘检测和直线检测等。       
OpenCV官网：https://opencv.org/       
OpenCV Github：https://github.com/opencv/opencv      
OpenCV 扩展算法库：https://github.com/opencv/opencv_contrib

#### 2.2 数据扩增（Data Augmentation）

在深度学习中数据扩增方法非常重要，数据扩增可以增加训练集的样本，同时也可以有效缓解模型过拟合的情况，也可以给模型带来的更强的泛化能力。

数据扩增方法有很多：从颜色空间、尺度空间到样本空间，同时根据不同任务数据扩增都有相应的区别。        
对于图像分类，数据扩增一般不会改变标签；对于物体检测，数据扩增会改变物体坐标位置；对于图像分割，数据扩增会改变像素标签。     

##### 2.2.1 常用的数据扩增库     

- #### torchvision      

  https://github.com/pytorch/vision      
  pytorch官方提供的数据扩增库，提供了基本的数据数据扩增方法，可以无缝与torch进行集成；但数据扩增方法种类较少，且速度中等；       

- #### imgaug         

  https://github.com/aleju/imgaug      
  imgaug是常用的第三方数据扩增库，提供了多样的数据扩增方法，且组合起来非常方便，速度较快；      

- #### albumentations       

  https://albumentations.readthedocs.io      
  是常用的第三方数据扩增库，提供了多样的数据扩增方法，对图像分类、语义分割、物体检测和关键点检测都支持，速度较快。      

#### 2.3 Pytorch读取数据 

在Pytorch中数据是通过Dataset进行封装，并通过DataLoder进行并行读取。

我们在定义好的Dataset基础上构建DataLoder，你可以会问有了Dataset为什么还要有DataLoder？其实这两个是两个不同的概念，是为了实现不同的功能。                 

- Dataset：对数据集的封装，提供索引方式的对数据样本进行读取      
- DataLoder：对Dataset进行封装，提供批量读取的迭代读取    

在加入DataLoder后，数据按照批次获取，每批次调用Dataset读取单个样本进行拼接（　<u>**这一步没看懂**</u>　）。此时data的格式为：       
                ``` torch.Size([10, 3, 64, 128]), torch.Size([10, 6]) ```          
前者为图像文件，为batchsize * chanel * height * width次序；后者为字符标签。      