import torch as tc
import Library.MathFun as mf
from matplotlib import pyplot as plt
from Library.MatrixProductState \
    import full_tensor, mps_norm_square


length, d, chi = 6, 2, 4
dtype = tc.float64

tensors = [tc.randn(1, d, chi, dtype=dtype)] + \
          [tc.randn(chi, d, chi, dtype=dtype)
           for _ in range(length-2)] + \
          [tc.randn(chi, d, 1, dtype=dtype)]

psi = tensors[0]
print('第0个张量的形状为：', tensors[0].shape)
for n in range(1, len(tensors)):
    print('第%d个张量的形状为：' % n, tensors[n].shape)
    psi = tc.tensordot(psi, tensors[n], [[-1], [0]])
    print('收缩完该张量后，所得张量的形状为：', psi.shape)
psi = psi.squeeze()
print('去掉前后维数为1的指标得到最终的高阶张量，形状为：')
print(psi.shape)

print('--------------------- 分割线 ---------------------')
norm_t = psi.norm()
norm_mps = mps_norm_square(tensors, True, form='norm')
print('高阶张量的模方 = %g' % norm_t**2)
print('MPS的模方 = %g' % norm_mps)
print('注：求MPS模方的同时已对MPS进行了归一化')

print('--------------------- 分割线 ---------------------')
psi /= norm_t
tensors1, lm1 = mf.ttd(psi)
print('对所得高阶张量进行TT分解，且不裁剪，误差为：')
psi1 = full_tensor(tensors1)
print(((psi1 - psi).norm()).item())

cut_off = 3
tensors2, lm2 = mf.ttd(psi, cut_off)
print('对所得高阶张量进行TT分解，虚拟指标维数裁剪为2，误差为：')
psi2 = full_tensor(tensors2)
print(((psi2 - psi).norm()).item())


