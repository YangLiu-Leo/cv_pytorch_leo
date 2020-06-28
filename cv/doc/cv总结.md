目前，公认的计算机视觉bai三大会议分别为duICCV,ECCV,CVPR。

ICCV的全称是 IEEE International Conference on Computer Vision，国际计算机视觉大会，两年召开一次

ECCV的全称是Europeon Conference on Computer Vision，两年一次，欧洲

CVPR的全称是Internaltional Conference on Computer Vision and Pattern Recogintion，视觉与模式识别，两年一次，美国

#### **1.视觉语言导航综述Visual Language Navigation （vln_algorithm）**

https://blog.csdn.net/cww97/article/details/103941518

既然要对自然语言的指令进行理解，那么逃不掉LSTM；对于图像信息的处理必然逃不开CNN[@NIPS2012_4824]；另外导航问题本质上是一个机器人控制问题，涉及决策，逃不开的便是强化学习的框架[@francoislavet2018introduction]。这三者的结合构成了当前几乎所有视觉语言导航任务乃至模态融合任务的基本结构

观察下来所有的方法都没有逃开seq-to-seq的基本架构，这是由该问题的输入输出决定的。但是在这个基本架构上针对这个问题还是可以做一些优化的。另外本文提及的所有方法对算法做的改进大致可以分为三类：

    对齐：包括轨迹于指令对齐、任务进度的对齐
    
    数据增强：使用自监督半监督发方法
    
    强化学习：结合model-free与model-based、结合模仿学习
另外对于整个视觉语言导航任务数据集的发展而言，人工标注的自然语言永远是最自然的，模拟器在一步步的往逼真度越来越高的靠。这个工作变得越来越costly。那么我们如何生成接近自然语言的数据标签，成为这种问题的一个新的方向，能否用自监督的方法去生成，自监督学习在CV领域已经有了很多的工作，然而在NLP领域目前并不多，主要由于先前需求不够硬且效果没有监督学习的好，面对构造数据集的代价越来越大、且无标签数据的没有代价，自监督一定是一个未来。

CVPR论文：https://blog.csdn.net/dQCFKyQDXYm3F8rB0/article/details/88148505

#### **2.数据集汇总**

 https://www.datasetlist.com/

https://blog.csdn.net/m0_38106923/article/details/89052216

https://blog.csdn.net/lingpy/article/details/79918345



#### 3.学习资料 深度之眼的 西瓜书 花书 CV NLP论文带读

cv研究和应用方向简介（有论文的arxiv地址）：https://blog.csdn.net/qq_25737169/article/details/80099628

机器学习与计算机视觉大牛族谱：http://blog.sina.com.cn/s/blog_6151255a01015tay.html

**计算机视觉牛人博客和代码汇总**：https://www.cnblogs.com/findumars/p/5009003.html

**cvpr2019论文汇总**：https://blog.csdn.net/qq_41895190/article/details/90371716

**cvpr2019**:https://blog.csdn.net/Extremevision/article/details/92066746

cv机器学习面试总结：https://blog.csdn.net/e01528/article/details/89221713

论文汇总：https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/86653489

RPN(RegionProposal Network)区域生成网络，two stage 结构Faster-RCNN的核心：https://blog.csdn.net/qq_36269513/article/details/80421990



tar zcvf xvf：https://blog.csdn.net/qian19950120/article/details/78122385

linux文件权限命名：https://blog.csdn.net/BjarneCpp/article/details/79912495

vim下的移动光标到指定行，复制剪切粘贴：https://blog.csdn.net/lixinghua666/article/details/82289809

http代理服务器：https://blog.csdn.net/JAck_chen0309/article/details/104781600

罗盘时钟：https://blog.csdn.net/qiqiyingse/article/details/90440377



