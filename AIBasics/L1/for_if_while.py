# for循环
# 注意range函数的应用
import random

start_num = 1
end_num = 9
change = 2

print('1到9（不包括9），每隔2取一次值')
for n in range(start_num, end_num, change):
    print(n)

print('0到9（不包括9），每隔1取一次值')
for m in range(end_num):
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

# while条件
y = 0.5
while y > 0.01:
    y = y ** 2
    print(y)


'''
作业：参考如下代码
for可以依次取出循环体中的各个值，例如列表属于循环体，
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
'''





