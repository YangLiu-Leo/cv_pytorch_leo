# cv_learn

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

[baseline]: https://github.com/datawhalechina/team-learning/tree/master/03%20%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%	"课程内容、任务及baseline"



#### 2.1 虚拟环境及依赖包的安装，及baseline的运行步骤

１）可以先创建虚拟环境，然后执行下面指令复现环境

pip install -r requirements.txt

２）download_dataset.py　可以从云端下次数据集

３）json_load.py　可以读一张图片然后根据json文件里的label把数字扣出来

４）baseline.py　可以训练模型，然后输出识别结果到.csv文件(可以WPS打开)里

#### 2.2 pytorch源码学习

Pytorch Sampler详解(界面动画很赞)：　https://www.cnblogs.com/marsggbo/p/11541054.html  

DataLoader源码阅读：　https://blog.csdn.net/u012436149/article/details/78545766 

#### 2.3 解题思路

赛题本质是分类问题，需要对图片的字符进行识别。本次赛题的难点是需要对不定长的字符进行识别，与传统的图像分类任务有所不同。

**简单入门思路：**定长字符识别，填充字符长度到定长

**专业字符识别思路：**不定长字符识别，比较典型的有CRNN字符识别模型，可以视为单词或句子

**专业分类思路：**检测再识别  

在赛题数据中已经给出了训练集、验证集中所有图片中字符的位置，因此可以首先将字符的位置进行识别，利用物体检测的思路完成。此种思路需要参赛选手构建字符检测模型，对测试集中的字符进行识别。选手可以参考物体检测模型SSD或者YOLO来完成。 