# 较为复杂的数学运算需要用到函数库，使用import命令导入
# 常用数学函数库：torch、math、numpy等

import numpy as np  # as：取别名
import math


# 例：指数函数 exp(x)
x = 0.8
print('exp(x) = ')
print(np.exp(x))
print(math.exp(x))

# 例：阶乘n!
print('4! = ', math.factorial(4))

"""
练习：
计算圆周率的自然对数 ln(pi)，打印出计算结果
"""

