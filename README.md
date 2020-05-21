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



### 二　天池不定长字符识别赛（char_detection文件夹里）

赛题目的是识别图中的数字门牌号码

2.1 虚拟环境及依赖包的安装，及baseline的运行步骤

１）可以先创建虚拟环境，然后执行下面指令复现环境

pip install -r requirements.txt

２）download_dataset.py　可以从云端下次数据集

３）json_load.py　可以读一张图片然后根据json文件里的label把数字扣出来

４）baseline.py　可以训练模型，然后输出识别结果到.csv文件(可以WPS打开)里

