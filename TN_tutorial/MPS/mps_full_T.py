import torch as tc


length, d, chi = 4, 2, 3
tensors = [tc.randn(d, chi)] + \
          [tc.randn(chi, d, chi) for _ in range(length-2)] + \
          [tc.randn(chi, d)]

psi = tensors[0]
print('第0个张量的形状为：', tensors[0].shape)
for n in range(1, len(tensors)):
    print('第%d个张量的形状为：' % n, tensors[n].shape)
    psi = tc.tensordot(psi, tensors[n], [[-1], [0]])
    print('收缩完该张量后，所得张量的形状为：', psi.shape)


