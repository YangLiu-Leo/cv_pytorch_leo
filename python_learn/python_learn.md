task1 变量 运算符 数据类型 位运算
1.数据类型 int float bool
''' ''' 或者 """ """ 表示区间注释，在三引号之间的所有内容被注释
notebook里 !ls 表示执行shell指令
is, is not 对比的是两个变量的内存地址
==, != 对比的是两个变量的值
in, not in
print(type(1))  print(isinstance(1, int))
有时候我们想保留浮点型的小数点后 n 位。可以用 decimal 包里的 Decimal 对象和 getcontext() 方法来实现。
2.按位操作
~ & | 分别是按位 取反 与 或，十进制数字会转换为二进制进行操作

通过 <<，>> 快速计算2的倍数问题。
n << 1 -> 计算 n*2
n >> 1 -> 计算 n/2，负奇数的运算不可用
n << m -> 计算 n*(2^m)，即乘以 2 的 m 次方
n >> m -> 计算 n/(2^m)，即除以 2 的 m 次方
1 << n -> 2^n

3.leetcode练习题：

leetcode 习题 136. 只出现一次的数字

给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

尝试使用位运算解决此题。



task2 条件 循环
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

# AssertionError
3.
