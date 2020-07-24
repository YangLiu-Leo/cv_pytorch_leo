### task1 变量 运算符 数据类型 位运算

##### 1.数据类型 int float bool

''' ''' 或者 """ """ 表示区间注释，在三引号之间的所有内容被注释
notebook里 !ls 表示执行shell指令
is, is not 对比的是两个变量的内存地址
==, != 对比的是两个变量的值
in, not in
print(type(1))  print(isinstance(1, int))
有时候我们想保留浮点型的小数点后 n 位。可以用 decimal 包里的 Decimal 对象和 getcontext() 方法来实现。

##### 2.按位操作

~ & | 分别是按位 取反 与 或，十进制数字会转换为二进制进行操作

通过 <<，>> 快速计算2的倍数问题。
n << 1 -> 计算 n*2
n >> 1 -> 计算 n/2，负奇数的运算不可用
n << m -> 计算 n*(2^m)，即乘以 2 的 m 次方
n >> m -> 计算 n/(2^m)，即除以 2 的 m 次方
1 << n -> 2^n

##### 3.总结

1）is == 的区别

- is, is not 对比的是两个变量的内存地址
- ==, != 对比的是两个变量的值
- 比较的两个变量，指向的都是地址不可变的类型（str等），那么is，is not 和 ==，！= 是完全等价的。
- 对比的两个变量，指向的是地址可变的类型（list，dict等），则两者是有区别的。

2) 运算符的优先级

- 一元运算符优于二元运算符。例如`3 ** -2`等价于`3 ** (-2)`。
- 先算术运算，后移位运算，最后位运算。例如 `1 << 3 + 2 & 7`等价于 `(1 << (3 + 2)) & 7`。
- 逻辑运算最后结合。例如`3 < 4 and 4 < 5`等价于`(3 < 4) and (4 < 5)`。

##### 4.leetcode练习题：

leetcode 习题 136. 只出现一次的数字

给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

尝试使用位运算解决此题。

```
# 思路：遍历数组依次异或，位不一样时为1
"""
Input file
example1: [2,2,1]
example2: [4,1,2,1,2]

Output file
result1: 1
result2: 4
"""

class Solution:
    def singleNumber(self, nums) :
        one=0
        for num in nums:
            one=one^num
        return one
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        return reduce(lambda x, y: x ^ y, nums)
    
if __name__ == "__main__":
    result = Solution()
    nums = [2,3,4,5,3,4,2]
    print(result.singleNumber(nums))

#c++    
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int ret = 0;
        for (auto e: nums) ret ^= e;
        return ret;
    }
};

```



### task2 条件 循环

1.if - elif - else 
条件表达式用布尔操作符 and，or和not
temp = input('请输入成绩:')
source = int(temp)
if 100 >= source >= 90:
    print('A')
elif 90 > source >= 80:
    print('B')
elif 80 > source >= 60:
    print('C')
elif 60 > source >= 0:
    print('D')
else:
    print('输入错误！')
2.assert 
assert这个关键词我们称之为“断言”，当这个关键词后边的条件为 False 时，程序自动崩溃并抛出AssertionError的异常。
my_list = ['lsgogroup']
my_list.pop(0)
assert len(my_list) > 0

AssertionError

3.for循环

直接in拿出来的是元素，in range(len(member))拿出来的是index

```
for i in range(2, 9):  # 不包含9
range里面是整数范围
```

```
member = ['张三', '李四', '刘德华', '刘六', '周润发']
for each in member:
    print(each)

# 张三
# 李四
# 刘德华
# 刘六
# 周润发

for i in range(len(member)):
    print(member[i])
```

4.元组 列表 集合 字典

```
(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
[[10, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]   x[0][0] = 10
{1, 2, 3, 4, 5, 6}
{0: True, 3: False, 6: True, 9: False}
```

5.生成器generator yield

```
e = (i for i in range(10))  #e是一个迭代器
print(e)
# <generator object <genexpr> at 0x0000007A0B8D01B0>

print(next(e))  # 0
print(next(e))  # 1

for each in e:
    print(each, end=' ')

# 2 3 4 5 6 7 8 9
```

6.作业

```
  password = input('enter password:') #输入
  break continue pass 占位符号
  
  import random
  secret = random.randint(1, 10) #[1,10]之间的随机数
  
  ### plot打印数组x,y 绘制多条曲线并标记颜色和名字
# # 计算正弦和余弦曲线上的点的 x 和 y 坐标 
# x = np.arange(0,  3  * np.pi,  0.1) 
# y_sin = np.sin(x) 
# y_cos = np.cos(x)  
# # 建立 subplot 网格，高为 2，宽为 1  
# # 激活第一个 subplot
# #     plt.subplot(2,  1,  1)  
# # 绘制第一个图像 
# plt.plot(x, y_sin, color='green', label='Sine') 
# plt.title('Sine')  
# # 将第二个 subplot 激活，并绘制第二个图像
# #     plt.subplot(2,  1,  2) 
# plt.plot(x, y_cos, color='skyblue', label='Cosine') 
# plt.title('Cosine')  
# plt.legend() # 显示图例、
# # 展示图像
# plt.show()
  
  1、编写一个Python程序来查找那些既可以被7整除又可以被5整除的数字，介于1500和2700之间。
  res = [i for i in range(1500,2700) if i%7==0 and i%5==0]
  2.计算龟兔赛跑谁获胜
	兔子每超过乌龟s米，就会停下来休息s秒
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# train_df = pd.read_csv('myspace/train_set.csv', sep='\t', nrows=20)

#输入:

#输入只有一行，包含用空格隔开的五个正整数v1，v2，t，s，l，其中(v1,v2< =100;t< =300;s< =10;l< =10000且为v1,v2的公倍数)

#输出:

#输出包含两行，第一行输出比赛结果——一个大写字母“T”或“R”或“D”，分别表示乌龟获胜，兔子获胜，或者两者同时到达终点。

#第二行输出一个正整数，表示获胜者（或者双方同时）到达终点所耗费的时间（秒数）。

# v1,v2,t,s,l = map(int,input("input:  v1,v2,t,s,l ").split())
# v1,v2,t,s,l = 1,2,3,2,100
v1=2
v2=1
t=10
s=8
l=50

# t_com是当前时间 每个循环+1; t_sum记录i每次兔子休息了的时间
t_com = 0
t_sum = []
rab_run = True
# rab = v1*t_com
# tor = v2*t_com
t_plot = []
rab = []
tor = []

while(v2*t_com < l):


    if rab_run:
        #### 计算兔子累计休息的时间，要记得每次都先清0
        sum_rest = 0
        for t in t_sum:
                sum_rest += t
#         print(t_com,sum_rest,t_com-sum_rest)
        rab_tmp = v1*(t_com - sum_rest)
        if(rab_tmp>=l):
            print('rabbit win')
            break
         # 兔子在跑 且 路程大于乌龟s
        if(rab_tmp > t+v2*t_com):
            rab_run = False
            t_stop=t_com ###记录一次兔子停下的时间
 
    # 兔子停下 且 休息了s
    if not rab_run and t_com - t_stop > s:
        rab_run = True
        t_sum.append(t_com-t_stop) # 把兔子本次休息了的时间记录到数组里
#         print(t_com,t_stop,t_com-t_stop)
        
    t_com+=1

    #开始画图
    t_plot.append(t_com)
    rab.append(rab_tmp)
    tor.append(v2*t_com)

### 法一 scatter持续画点，构成两条曲线
    plt.title("x-t.") 
#     plt.annotate("(3,6)", xy = (3, 6), xytext = (4, 5), arrowprops = dict(facecolor = 'black', shrink = 0.1))
    plt.xlabel("t")
    plt.ylabel("x")
    plt.scatter(t_com, v2*t_com, color='green', label='rabbit')
    plt.scatter(t_com, rab_tmp, color='skyblue', label='tortoise') ### plot 会把前面画上去的点清掉
    plt.plot()
#     plt.pause(0.1)         # 暂停一秒，下次画的点将出现在一个新窗口
#     plt.show() #感觉没啥用,而且下个循环画的点将出现在一个新窗口，是保证产生新的instance？


else:
    if(rab_tmp>=l):
        print('draw') #平局
    else:
        print('tortoise win')

### 法二 plot一次性画两条曲线
# plt.plot(t_plot, rab, color='green', label='rabbit') 
# plt.plot(t_plot, tor, color='skyblue', label='tortoise') 
# plt.title('x-t curve')  
plt.legend() # 显示图例、
# 展示图像
plt.show()
        

```



### task3 异常处理

C++里有try catch;

pyton里有try except，应用报错机制，如果代码抛出错误程序会停止运行并打印错误