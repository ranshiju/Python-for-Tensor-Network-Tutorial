def add(x, y):
    # 加法
    return x + y


print(1 + 2)
print(add(1, 2))


def remove_numbers(x:'str'):
    # 去掉字符串里的所有数字
    x1 = ''
    for s in x:
        if ord(s) < ord('0') or ord(s) > ord('9'):
            x1 += s
    return x1


def remove_numbers_best(x: 'str'):
    # 去掉字符串里的所有数字
    import re
    return re.sub(r'\d+', '', x)


test_str = 'h9e76l34l0o11'
print(remove_numbers(test_str))
print(remove_numbers_best(test_str))


'''
练习：
1. 将TaylorExpansion.py中的内容定义成一个函数，命名为exp_tl，
该函数输入为x的值与展开阶数，输出为exp(x)在对应阶数下的取值。
2. 在AI大模型的帮助下，编写一个函数，其功能是判断输入的正整数是否为质数，
尝试读懂函数代码并调用函数。
'''

