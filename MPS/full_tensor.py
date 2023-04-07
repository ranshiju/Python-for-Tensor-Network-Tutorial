import torch as tc
import Library.MatrixProductState as mps


print('建立开放边界MPS')
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
norm_t = psi.norm() ** 2
norm_mps, _ = mps.norm_square(tensors, False, form='norm')
print('高阶张量的模方 = %g' % norm_t)
print('MPS的模方 = %g' % norm_mps)

print('--------------------- 分割线 ---------------------')
print('建立周期边界MPS')
tensors1 = mps.random_mps(length, d, chi, 'periodic')

psi1 = tensors1[0]
print('第0个张量的形状为：', tensors1[0].shape)
for n in range(1, len(tensors1)):
    print('第%d个张量的形状为：' % n, tensors1[n].shape)
    psi1 = tc.tensordot(psi1, tensors1[n], [[-1], [0]])
    print('收缩完该张量后，所得张量的形状为：', psi1.shape)
print('根据周期边界调节，将所剩两个虚拟指标求和')
psi1 = psi1.permute([0, length+1] + list(range(1, length+1))
                    ).reshape(chi, chi, -1)
psi1 = tc.einsum('aab->b', psi1).reshape([d] * length)
print('得到的高阶张量形状为：')
print(psi1.shape)

psi2 = mps.full_tensor(tensors1)
err = (psi1 - psi2).norm().item()
print('循环与内置函数所得全局张量相差：%g' % err)

norm1, _ = mps.norm_square(tensors1, False, form='norm')
print('周期MPS模方为：%g' % norm1.item())
print('全局张量的模方为：%g' % psi2.norm() ** 2)
