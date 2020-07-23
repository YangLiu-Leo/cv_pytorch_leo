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
  
  1、编写一个Python程序来查找那些既可以被7整除又可以被5整除的数字，介于1500和2700之间。
  res = [i for i in range(1500,2700) if i%7==0 and i%5==0]
```