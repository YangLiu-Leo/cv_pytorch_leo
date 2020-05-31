# DL-pytorch

```
1.第一期打卡链接　https://jinshuju.net/f/lU2pgj/r/LvWl2e
第二期打卡链接　https://jinshuju.net/f/drWQSp/r/ZHuLQD
```

#### 1.线性回归(用来预测连续值)　计算损失函数平均值

```
#q1:　+=　前面的变量要提前定义好
#q2: #注释

#定义损失函数：
import torch
def squared_loss(y_hat, y): 
    return (y_hat - y.view(y_hat.size())) ** 2 / 2
y_h = torch.tensor([2.33,1.07,1.23])
y = torch.tensor([3.14,0.98,1.32])

result = 0　　＃下面　+= 需要事先定义变量
result += squared_loss(y_h,y)
'mean loss: %.3f' % result.mean()
```

#### ２.softmax(也是线性单层模型，用来预测离散值，如图像分类)

```
softmax([100, 101, 102])
```

```
q1: .exp()不能处理LONG类型，要转换为flot(可以通过3.表示浮点型)

import torch
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    # print("X size is ", X_exp.size())
    # print("partition size is ", partition, partition.size())
    return X_exp / partition  # 这里应用了广播机制
x = torch.tensor([[1.,2,2],[2,3,5]],dtype=torch.float64)
res = softmax(x)
print(res)
```

#### 3.多层感知机(含隐藏层，但是联立后依旧等价于单层神经网络)

激活函数(全连接层间引入的非线性变换），ReLU较为常用；Sigmoid和tanh可以缩小数值的区间，但是可能会引起梯度消失，慎用！

下面是常用的激活函数：

ReLU: 小于0部分清零，大于0部分不变

Sigmoid: 变换到 0~1 之间

tanh(双曲正切): 变换到 -1~1 之间

```
#q1:tensor变量有什么函数，如x.type，x.shape[1](取一列),x.view()(resize形状)
#q2:print函数如何同时输出注释和变量　print('a:', a)

#定义模型参数：
import torch
pixel = 256*256
print(pixel)
num_inputs, num_outputs, num_hiddens = pixel, 10, 1000

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)
print("type:")
print(len(W1), type(W1), W1.type, W1.size())
print("all weight number:")
print((W1.size()[0]*W1.size()[1]+W2.size()[0]*W2.size()[1]))

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)
```

#### 4.文本预处理

构建字典{}

tokens可以是列表(list) [(),()]、元组(tuple) ( ,  ,)、字符串

#### ５.语言模型

这里介绍的是基于统计的语言模型，主要用n-gram(n元语法)，n-gram是基于n-1阶马尔可夫链简化的语言模型。后续会介绍基于神经网络的语言模型。

set()去重　join()拼接

批量大小batch_size是每个小批量的样本数

#### 6.循环神经网络（Recurrent Neural Network, RNN）

6.1 循环神经网络（Recurrent Neural Network, RNN）是一类以[序列](https://baike.baidu.com/item/序列/1302588)（sequence）数据为输入，在序列的演进方向进行[递归](https://baike.baidu.com/item/递归/1740695)（recursion）且所有节点（循环单元）按链式连接的[递归神经网络](https://baike.baidu.com/item/递归神经网络/16020230)（recursive neural network）.

双向循环神经网络（Bidirectional RNN, Bi-RNN）和长短期记忆网络（Long Short-Term Memory networks，[LSTM](https://baike.baidu.com/item/LSTM/17541102)）是常见的的循环神经网络.

6.2 卷积神经网络（Convolutional Neural Networks, CNN）是一类包含[卷积](https://baike.baidu.com/item/卷积/9411006)计算且具有深度结构的[前馈神经网络](https://baike.baidu.com/item/前馈神经网络/7580523)（Feedforward Neural Networks).

6.3 一些概念

加法的广播机制　仿射变换

.apend()

#### 7.过拟合、欠拟合及其解决方案

训练数据集(train)可以用来调整模型参数；测试数据集(test)；验证数据集

测试数据集不可以用来调整模型参数，如果使用测试数据集调整模型参数，可能在测试数据集上发生一定程度的过拟合，此时将不能用测试误差来近似泛化误差。



过拟合是指训练误差达到一个较低的水平，而泛化误差依然较大。
 欠拟合是指训练误差和泛化误差都不能达到一个较低的水平。
 发生欠拟合的时候在训练集上训练误差不能达到一个比较低的水平，所以过拟合和欠拟合不可能同时发生。

L2范数正则化也就是权重衰减是用来应对过拟合的。



过拟合除了增加增加训练数据集的大小，还可以使用权重衰减和丢弃法来缓解

#### 8.梯度消失、梯度爆炸

我们在模型选择、欠拟合和过拟合中介绍了*𝐾*折交叉验证。它将被用来选择模型设计并调节超参数。

模型训练实战步骤：

1. 获取数据集
2. 数据预处理
3. 模型设计
4. 模型验证和模型调整（调参）
5. 模型预测及提交

#### 9.循环神经网络进阶

门控循环神经网络GRU/LSTM,LSTM和GRU能一定程度缓解梯度消失与梯度爆炸的问题。

GRU有重置门和更新门，没有遗忘门。重置门有助于捕捉时间序列里短期的依赖关系，更新门有助于捕捉时间序列⾥长期的依赖关系。

深度循环神经网络：在pyorch模型建立时，设置上述模型的num_layers>1即可得到多层的循环神经网络，但并不是层数越多越好，因为层数越多模型越复杂且对数据集要求更高。

双向循环神经网络:双向循环神经网络在文本任务里能做到同时考虑上文和下文与当前词之间的依赖。前向的HtH_tHt和后向的HtH_tHt用`concat`进行连结.

#### 10.机器翻译(MT)及相关技术

Encoder-Decoder

Sequence to Sequence模型

集束搜索(Beam Search)结合了greedy search和维特比算法，集束搜索是维特比算法的贪心形式，得到的不是全局最优解？。

数据预处理中分词(Tokenization)的工作是把字符形式的句子转化为单词组成的列表。

单词转化为词向量是模型结构的一部分，词向量层一般作为网络的第一层。

Encoder-Decoder常应用于输入序列和输出序列的长度是可变的，如选项一二四，而分类问题的输出是固定的类别，不需要使用Encoder-Decoder

#### 11.注意力机制与Seq2seq模型

注意力机制借鉴了人类的注意力思维方式，以获得需要重点关注的目标区域。在Dot-product(点积) Attention中，key与query维度需要一致，在(多层感知机)MLP Attention中则不需要。

注意力掩码可以用来解决一组变长序列的编码问题。有效长度不同导致 Attention Mask 不同，屏蔽掉无效位置后进行attention，会导致不同的输出。

#### 12.Transformer

Seq2seq with Attention 

Transformer：

Decoder 部分的第二个注意力层不是自注意力，key-value来自编码器而query来自解码器；自注意力模块理论上可以捕捉任意距离的依赖关系；训练过程解码器部分只需进行一次前向传播，预测过程要进行句子长度次。

批归一化（Batch Normalization）才是对每个神经元的输入数据以mini-batch为单位进行汇总；而层归一化不是。

在Transformer模型中，注意力头数为h，嵌入向量和隐藏状态维度均为d，那么一个多头注意力层所含的参数量是？

h个注意力头中，每个的参数量为3d2，最后的输出层形状为hd×d，所以参数量共为4hd2。



02/18科研讲座

向外行一样思考，向专家一样执行。

人工智能发展的两个方向：深度学习、逻辑推理，目前逻辑推理方面发展较弱，接下来的发展方向就是二者结合。

泛化：举一反三。统计泛化、组合泛化。

自然语言处理、知识图谱。语义信息、语言结构。nlp热门的方向基于上下文的向量表示方法。



#### 13.卷积神经网络基础(CNN)



如果原输入的高和宽是*𝑛**ℎ*和*𝑛**𝑤*，卷积核的高和宽是*𝑘**ℎ*和*𝑘**𝑤*，在高的两侧一共填充*𝑝**ℎ*行，在宽的两侧一共填充*𝑝**𝑤*

列，则输出形状为：

(*𝑛**ℎ*+*𝑝**ℎ*−*𝑘**ℎ*+1)×(*𝑛**𝑤*+*𝑝**𝑤*−*𝑘**𝑤*+1)

我们在卷积神经网络中使用奇数高宽的核，比如3×3

，5×5的卷积核，对于高度（或宽度）为大小为2*𝑘*+1的核，令步幅为1，在高（或宽）两侧选择大小为*𝑘*的填充，便可保持输入与输出尺寸相同。



互相关运算、卷积运算

全连接层、卷积层、池化层

`1.假如你用全连接层处理一张256×256的彩色（RGB）图像，输出包含1000个神经元，在使用偏置的情况下，参数数量是：`

图像展平后长度为3×256×256，权重参数和偏置参数的数量是3×256×256×1000+1000=196609000。

2.假如你用全连接层处理一张256×256的彩色（RGB）图像，卷积核的高宽是3×3，输出包含10个通道，在使用偏置的情况下，这个卷积层共有多少个参数：

输入通道数是3，输出通道数是10，所以参数数量是10×3×3×3+10=280。

`３．conv2d = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=2)`，输入一张形状为3×100×100的图像，输出的形状为：

4×102×102。输出通道数是4，上下两侧总共填充4行，卷积核高度是3，所以输出的高度是104−3+1=102，宽度同理可得。

４．1×1卷积可以看作是通道维上的全连接

卷积层通过填充、步幅、输入通道数、输出通道数等调节输出的形状

两个连续的3×3卷积核的感受野与一个5×5卷积核的感受野相同



#### 14.LeNet

LeNet主要分为两个部分：卷积层块和全连接层块;

LeNet模型中，90%以上的参数集中在全连接层块，全连接层有局限性;

LeNet在连接卷积层块和全连接层块时，需要做一次展平操作;

LeNet的卷积层块交替使用卷积层和池化层。



使用形状为2×2，步幅为2的池化层，会将高和宽都减半

卷积神经网络通过使用滑动窗口在输入的不同位置处重复计算，减小参数数量

在通过卷积层或池化层后，输出的高和宽可能减小，为了尽可能保留输入的特征，我们可以在减小高宽的同时增加通道数

#### 15.AlexNet

换成ReLU函数稀疏化/正则化

Dropout提高泛化能力／缓解过拟合情况



使用重复元素的网络VGG(可以重复堆叠VGG block)

参数多样本少容易过拟合

网络中的网络NiN

GoogLeNet



#### 16.批量归一化(BatchNormalization)和残差网络(ResNet)

bn 　　bp(反向传播)　　fc(全连接)

稠密连接网络(DenseNet)

#### 17.凸优化　凸性 （Convexity）

1).定义

凸函数：两点连线上的点的数值　大于该点对应的函数值

Jensen不等式：函数值的期望　大于　期望的函数值

凸函数的性质：无局部最小值（反证法）；与凸集的关系；凸函数与二阶导数大于0是充要条件

2).

优化在深度学习中面临的挑战: 局部最小值, 鞍点, 梯度消失

鞍点是对所有自变量一阶偏导数都为0，且Hessian矩阵特征值有正有负的点

3).

有限制条件的优化问题可以用以下方法解决：

拉格朗日乘子法, 添加惩罚项, 投影法

#### 18.梯度下降

1).常规方法

梯度下降: 梯度下降是沿着梯度的反方向移动自变量从而减小函数值的。

是对所有的样本进行计算梯度更新梯度（所需epoch多）

2).随机梯度下降

随机梯度下降: 是每次对单一样本进行计算梯度（每个epoch用时多）

小批量随机梯度下降: 是每次对部分参数进行计算梯度

三个超参数：学习率，batch_size，epoch

3).自适应方法

牛顿法(引入了二阶倒数，但是对有局部最小值情况当学习率不合适的时候仍然会震荡)

动态学习率(加速收敛，解决收敛点抖动)

动量(解决局部最小值问题)

#### 19.optimization advance

针对两种 ill-conditioned Problem,有以下解决办法：

1).动量法  指数加权移动平均EMA(Exponential Moving Average)

如Momentum Algorithm里用到

2).Preconditioning gradient vector, 自适应学习率

用于rescale不同维度的梯度的数量级，以便于统一使用较大学习率

如AdaGrad　RMSProp里用到



Adam是RMSProp和Momentum算法的结合，并对EMA权重进行了无偏操作

Adam算法中的 mt和 vt (原文符号)分别是梯度的一阶矩和二阶矩估计，二者相比，可以使更新量rescale到1的附近。

#### 20.word2vec　词嵌入基础

one-hot向量表示单词，实现简便但没有体现词之前的相似度

余弦相似度表示词，可以表达词之间的相似度



#### 21.词嵌入进阶

词嵌入模型的训练本质上是在优化模型预测各词语同时出现的概率

抽象地说，词嵌入方法都是通过在大规模的语料库上进行训练，来让模型更好地“理解”词义，而好的模型设计则能提高训练的效率及模型的上界

由于他人训练词向量时用到的语料库和当前任务上的语料库通常都不相同，所以词典中包含的词语以及词语的顺序都可能有很大差别，此时应当根据当前数据集上词典的顺序，来依次读入词向量，同时，为了避免训练好的词向量在训练的最初被破坏，还可以适当调整嵌入层的学习速率甚至设定其不参与梯度下降

在进行预训练词向量的载入时，我们需要根据任务的特性来选定语料库的大小和词向量的维度，以均衡模型的表达能力和泛化能力，同时还要兼顾计算的时间复杂度