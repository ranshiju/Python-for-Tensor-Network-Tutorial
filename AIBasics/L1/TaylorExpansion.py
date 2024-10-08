import numpy as np


# 例：泰勒展开求指数 exp(x)
x = 0.8
order = 8  # 总阶数
yn = 1  # 用于储存第n个求和项
y = 1  # 用于储存求和计算结果（初始化为0阶项的值 ）
for n in range(1, order+1):
    yn = yn * x / n
    y = y + yn
    print('计算到第%i阶时，所得值为%.10g，误差为%.10g' % (n, y, abs(y-np.exp(x))))


"""
作业：
1. 仿照上述exp函数的泰勒展开代码，计算sin(x)的泰勒展开（计算到11阶）
2. 在python中，圆周率的值可由np.pi调出，利用print函数与‘%’，打印圆周率到小数点后第10位
"""
