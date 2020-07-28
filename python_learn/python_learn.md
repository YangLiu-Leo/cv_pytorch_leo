

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



### task4 列表 元组 字符串

# 

简单数据类型

- 整型`<class 'int'>`
- 浮点型`<class 'float'>`
- 布尔型`<class 'bool'>`

容器数据类型

- 列表`<class 'list'>`
- 元组`<class 'tuple'>`
- 字典`<class 'dict'>`
- 集合`<class 'set'>`
- 字符串`<class 'str'>`



列表不像元组，列表内容可更改 (mutable)，因此附加 (`append`, `extend`)、插入 (`insert`)、删除 (`remove`, `pop`) 这些操作都可以用在它身上。

#### 1.添加元素

`list.append(obj)` 在列表末尾添加新的对象，只接受一个参数，参数可以是任何数据类型，被追加的元素在 list 中保持着原结构类型。

`list.extend(seq)` 在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。

`list.insert(index, obj)` 在编号 `index` 位置插入 `obj`。

#### 2.移除元素

`list.remove(obj)` 移除列表中某个值的第一个匹配项
`list.pop([index=-1])` 移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。0表示第一位元素，-1表示倒数第一位元素
`del x[0:2]` 从列表中删除一个元素，且不再以任何方式使用它

```

x = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
x.remove('Monday')
print(x)  # ['Tuesday', 'Wednesday', 'Thursday', 'Friday']

y = x.pop()
print(y)  # Friday

y = x.pop(0)
print(y)  # Monday

y = x.pop(-2)
```

#### 3.获取列表中的元素

- 情况 4 - "start : stop : step"
- 以具体的 `step` 从编号 `start` 往编号 `stop` 切片。注意最后把 `step` 设为 -1，相当于将列表反向排列。

【例子】

```
week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
print(week[1:4:2])  # ['Tuesday', 'Thursday']
print(week[:4:2])  # ['Monday', 'Wednesday']
print(week[1::2])  # ['Tuesday', 'Thursday']
print(week[::-1])  
# ['Friday', 'Thursday', 'Wednesday', 'Tuesday', 'Monday']
```

- 情况 5 - " : "
- 复制列表中的所有元素（浅拷贝）。

【例子】

```
week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
print(week[:])  #切片操作拷贝，但依然不算深拷贝，只会拷贝第一层
list1 = week
# ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

list1 = [1,2,[3,4]]
list2 = list1[:]
list2[0]=0
list2[2][0]=8
print list1  # [1, 2, [8, 4]]


list2 = week #浅拷贝
week.sort() #浅拷贝，如果改变list2 week也会改变，两个一起改变。为了保证A B不被一起修改，这里要用深拷贝deepcopy
id(list2) #可以看到地址

```

**浅拷贝的意思:**
**如果p->a; q=p; 则q->a**

**c++涉及指针复制的时候要用深拷贝；python = 给list变量赋值时不要用浅拷贝**



#### 4.列表的其他方法

`list.count(obj)` 统计某个元素在列表中出现的次数

【例子】

```
list1 = [123, 456] * 3
print(list1)  # [123, 456, 123, 456, 123, 456]
num = list1.count(123)
print(num)  # 3
```

`list.index(x[, start[, end]])` 从列表中找出某个值第一个匹配项的索引位置

【例子】

```
list1 = [123, 456] * 5
print(list1.index(123))  # 0
print(list1.index(123, 1))  # 2
print(list1.index(123, 3, 7))  # 4
```

`list.sort(key=None, reverse=False)` 对原列表进行排序。

`key` -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。

`reverse` -- 排序规则，`reverse = True` 降序， `reverse = False` 升序（默认）。

【例子】

```
# 获取列表的第二个元素
def takeSecond(elem):
    return elem[1]


x = [(2, 2), (3, 4), (4, 1), (1, 3)]
x.sort(key=takeSecond)
print(x)
# [(4, 1), (2, 2), (1, 3), (3, 4)]

x.sort(key=lambda a: a[0]) #传入list a，返回a[0]
print(x)
# [(1, 3), (2, 2), (3, 4), (4, 1)]
```

#### 作业

1.列表操作

```
1. 在列表的末尾增加元素15
2. 在列表的中间位置插入元素20
3. 将列表[2, 5, 6]合并到lst中
4. 移除列表中索引为3的元素
5. 翻转列表里的所有元素
6. 对列表里的元素进行排序，从小到大一次，从大到小一次

lst = [2, 5, 6, 7, 8, 9, 2, 9, 9]
lst.append(15)
print lst

lst.insert((int)(len(lst)/2),20)
print lst

lst.extend([2,3,5])
print lst

lst.pop(3)
print lst

lst.reverse()
print lst

lst.sort()
print lst

lst.sort(reverse = True)
print lst
```



```
'''
2、修改列表
问题描述：
lst = [1, [4, 6], True]
请将列表里所有数字修改成原来的两倍
'''

lst = [1, [4, 6], True]

def double_lst(l):
  for i in range(len(l)):
	if type(l[i]) is int:
		l[i] <<= 1
	elif type(l[i]) is list:
		double_lst(l[i])
  return l

b = double_lst(lst)
print b
```

```
#### 852. 山脉数组的峰顶索引](https://leetcode-cn.com/problems/peak-index-in-a-mountain-array/)

class Solution:
  def peakIndexInMountainArray(self, A: List[int]) -> int:
​    B = A[:]  #为了保证A B不被一起修改，这里要用深拷贝deepcopy或者切片操作
​    B.sort()
​    return A.index(B[-1])

test = [0,2,1,0]
res = Solution()
print (res.peakIndexInMountainArray(test)) 
```



### 5.元组

- Python 的元组与列表类似，不同之处在于tuple被创建后就不能对其进行修改，类似字符串。
- 元组使用小括号，列表使用方括号。
- 元组与列表类似，也用整数来对它进行索引 (indexing) 和切片 (slicing)。



元组大小和内容都不可更改，因此只有 `count` 和 `index` 两种方法。

【例子】

```
t = (1, 10.31, 'python')
print(t.count('python'))  # 1
print(t.index(10.31))  # 1
```

### 6.作业

```
1.元组概念
写出下面代码的执行结果和最终结果的类型

(1, 2)*2  #(1, 2, 1, 2) <type 'tuple'>
(1, )*2  #(1, 1) <type 'tuple'>
(1)*2  #2 <type 'int'>
```

```
2. 以下代码属于从元组中拆包
a, b = 1, 2
```

leetcode 5题 最长回文子串



immutable object(str, int, tuple, dict's key)，if modified value, will creat a new object

mutable object(list, dict, set), variables will change

```
(mutable oprations like y.append(10) and y.sort(), but y =y+[10] and sorted(y) creat a new object)
```

也就是说，不可变类型 a=2  b=a。 首先 b也会存储a的地址；如果修改a=4，a会重新指向一块地址；如果修改b，b也会重新指向一块地址。所以=操作后，两个变量是独立互不干扰的

但是对于不可变类型 lst1=[1,2,3]  lst2=lst1。无论修改lst1还是lst2，二者都是指向一块地址