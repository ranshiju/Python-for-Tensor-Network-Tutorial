import torch as tc


x0 = tc.randn(3, 3, 4)
x1 = tc.randn(3, 4, 5)
x2 = tc.randn(3, 5, 3)

print('使用einsum函数计算张量收缩：')
y = tc.einsum('iab,jbc,kca->ijk', x0, x1, x2)

print('使用矩阵乘法计算上述结果：')
tmp = x0.reshape(-1, 4).mm(x1.permute(1, 0, 2).reshape(4, -1))
print('中间变量的形状为：', tmp.shape)
tmp = tmp.reshape(3, 3, 3, 5)
tmp = tmp.permute(0, 2, 1, 3).reshape(9, -1).mm(x2.permute(2, 1, 0).reshape(-1, 3))
print('中间变量的形状为：', tmp.shape)
z = tmp.reshape(3, 3, 3)

print('计算两种方式所得张量相减的L2范数：')
print((y - z).norm())

'''
练习：
1. 思考并验证：如果将第9行中的字符串公式'iab,jbc,kca->ijk'改为
                      'vqw,kwf,hfq->vkh'
  计算结果是否会改变？（注：理解哑指标的命名与计算结果无关）
2. 分别利用einsum函数与矩阵乘法，计算PPT中所给图形表示（作业图1）的张量求和结果，
  其中，每个张量的元素均由randn函数生成，且每个指标的维数均为2。
3. 定义超对角单位张量，其每个指标的维数必须相等，仅当所有指标取相等的值时，对应的张
  量元素为1，否则张量元素值为0。要求定义python函数，其输入为阶数与每个指标的维数，
  返回对应的超对角单位张量。
4. 编程证明PPT图（作业图2）所示等式，图中除超对角单位张量外，其余张量均为randn生成
  的张量，且每个指标维数为2。
'''

