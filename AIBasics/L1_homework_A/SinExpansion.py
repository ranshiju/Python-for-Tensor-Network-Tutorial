"""
作业：1. 仿照上述exp函数的泰勒展开代码，计算sin(x)的泰勒展开（计算到11阶）
"""

import numpy as np


x = 0.8
order = 11  # 总阶数
yn = x  # 用于储存第n个求和项
y = x  # 用于储存求和计算结果（初始化为0阶项的值 ）
for n in range(3, order+1, 2):
    yn = yn * (-1) * (x ** 2) / (n * (n-1))
    y = y + yn
    print('计算到第%i阶时，所得值为%.10g，误差为%.10g' % (n, y, abs(y-np.sin(x))))


"""
作业：2. 在python中，圆周率的值可由np.pi调出，利用print函数与‘%’，打印圆周率到小数点后第10位
"""

print('%.11g' % np.pi)
