import json
import numpy as np
import cv2
from matplotlib import pyplot as plt 
train_json = json.load(open('dataset/mchar_train.json'))

#数据标注处理
def parse_json(d):
    arr = np.array([d['top'],d['height'],d['left'],d['width'],d['label']])
    arr = arr.astype(int)
    return arr

img = cv2.imread('dataset/mchar_train/000008.png')

arr = parse_json(train_json['000008.png'])
#print(arr)
plt.figure(figsize=(10,10))
plt.subplot(1, arr.shape[1]+1,1)
plt.imshow(img)
plt.xticks([])
plt.yticks([])
print('char num: ', arr.shape[1])

for idx in range(arr.shape[1]):
   plt.subplot(1, arr.shape[1]+1, idx+2)
   plt.imshow(img[arr[0, idx]:arr[0, idx]+arr[1, idx],arr[2, idx]:arr[2, idx]+arr[3, idx]])
   plt.title(arr[4, idx])
   plt.xticks([]); plt.yticks([])
#plt.show()

