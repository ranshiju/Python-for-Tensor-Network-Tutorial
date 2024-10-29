import random

print('1到9（不包括9），每隔2取一次值')
for n in [1, 3, 5, 7]:
    print(n)

for n in range(1, 9, 2):
    print(n)

print('0到9（不包括9），每隔1取一次值')
for m in range(9):
    print(m)

# if判断
x = random.random()
print(x)
if x >= 0.5:
    print('随机得到的x >= 0.5')
elif x < 0.1:
    print('随机得到的x < 0.1')
else:  # 其他情况（上面的条件均不满足）
    print('随机得到的x属于(0.1, 0.5]')

print(1 < 0.5)  # 布尔变量

# while条件
y = 1
s = list()  # 用于储存数列
while y < 1024:
    y = y * 2
    print(y)
    s.append(y)
print(s)

'''
练习：
1. 参考如下代码：for可以依次取出循环体中的各个值，
例如：取出列表中的字符串
'''
x = [1, 'a', 3.3, 'haha']
y = list()  # 建立一个空的列表
for k in x:
    if type(k) is str:
        y.append(k)
print('取出的字符串有：', y)

'''
已知字符串本身也属于循环体，考虑如下字符串：
x = '22sffr3d44fcd32'
利用程序取出其中的阿拉伯数字字符，将其存到一个列表中。
（提示：利用ord()函数）

2. 分别利用range函数与while语句，生成从2到10的等差数列，间隔为2
（生成结果储存到一个列表中，并打印出来）
'''





