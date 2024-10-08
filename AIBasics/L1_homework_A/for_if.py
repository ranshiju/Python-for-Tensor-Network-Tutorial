"""
作业：已知字符串本身也属于循环体，考虑如下字符串：
x = '22sffr3d44fcd32'
利用程序取出其中的阿拉伯数字字符，将其存到一个列表中。
（提示：利用ord()函数）
"""

x = '22sffr3d44fcd32'

print(ord('0'), ord('9'))
y = list()
for k in x:
    if (ord(k) >= ord('0')) and (ord(k) <= ord('9')):
        y.append(k)
print(y)

