class student:

    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.if_healthy = True
        print('建立student实例：', name)

    def basic_info(self):
        print('=' * 20)
        print('下面是我的基本信息：')
        print('我的名字是' + self.name)
        print('年龄为%i岁' % self.age)
        health = '健康' if self.if_healthy else '不健康'
        print('我现在的健康状态：' + health)
        print('=' * 20)

    def aging(self, year=1):
        print('我一下子长大了%i岁' % year)
        self.age += year
        print('现在的年龄为%i岁' % self.age)

    def rename(self, name):
        print('我改名了，原来叫' + self.name + '，现在改名为' + name)

    def got_sick(self):
        self.if_healthy = False
        print('我生病了，健康状态变为：' + '不健康')


S1 = student(age=12, name='大宝')
print('实例的类型为：', type(S1))
print('属性name为：', S1.name)

S1.basic_info()
S1.aging(3)
S1.got_sick()
S1.rename('大壮')
S1.basic_info()

"""
作业：
1. 通过网络资料，了解除__init__外其他至少一种魔法方法；
2. 在上述student类中，修改__init__函数，增加属性location，并通过__init__函数输入赋值；
3. 在上述student类中，增加方法travel，其输入为新的地址（字符串），该函数将更新属性location更
新为输入的字符串，并打印出更新的内容（具体语言表述可自行决定）。
4. （选做）通过网络资料，自学类的继承（不需要知道原理，只需掌握使用方法）。
"""

