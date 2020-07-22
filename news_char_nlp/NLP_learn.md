task1 赛题理解
1.数据集下载
报名后给出一个csv文件的下载权限，该文件包括file description size link四列，每行代表训练集，验证集等
可以手动访问link下载保存(最快)；也可以写python脚本下载保存(跑的时候报错，pandas utf-8编码错误)

import pandas as pd 
import os
import requests
import zipfile
import shutil
links = pd.read_csv('NLP_data_list_0715.csv')
print(links)
dir_name = 'dataset'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
for i,link in enumerate(links['link']):
    file_name = links['file'][i]
    print(file_name, '\t', link)
    file_name = dir_name+'/'+file_name
    if not os.path.exists(file_name):
        response = requests.get(link, stream = True)
        with open(file_name ,'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

zip_list = ['train_set.csv', 'test_a.csv']

for little_zip in zip_list:
    if not os.path.exists(dir_name+'/'+little_zip):
        zip_file = zipfile.ZipFile(dir_name+'/'+little_zip+'.zip', 'r')
        zip_file.extractall(path=dir_name)


if os.path.exists(dir_name + '/' + '__MACOSX'):
    shutil.rmtree(dir_name + '/' + '__MACOSX')

2.notebook使用
注册了阿里云账号，进入天池专区，找到学习赛NLP比赛，然后论坛里会有base_line
可以fork base_line到我的实验室，点编辑即可申请kernal在线运行代码(一次2小时)，这里可以上传电脑本地的文件，也可以下载文件到本地

3.数据集分析
训练集有20w条新闻，每条新闻有两列即label text，然后可以通过pandas读入
统计label有多少种及其分布，总字数有多少及其分布，出现次数最多的某字符有可能是标点符合，根据标点符号分析一条新闻有几句话

4.数据统计及可视化分析

%pylab inline  可以在notebook中内嵌图片

!pwd & ls  可以看到当前notebook的路径及下面的文件

对新闻句子长度的统计可以得出，每个句子平均由多少个字符构成，最短的句子长度，最长的句子长度，并且可以直方图可视化分布；

对新闻类别统计，可以看出数据集中标签的分布，{'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6, '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}；

对字符分布统计，在训练集中总共包括6869个字，其中编号3750的字出现的次数最多，编号3133的字出现的次数最少。

5.作业

1. 假设字符3750，字符900和字符648是句子的标点符号，请分析赛题每篇新闻平均由多少个句子构成？

   判断每句话中有n个这样的标点字符，然后就有n+1句话

2. 统计每类新闻中出现次数对多的字符