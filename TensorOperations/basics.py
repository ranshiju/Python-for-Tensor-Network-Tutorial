import torch as tc

print('# 标量')
x = tc.tensor(1)
print('x = ', x)
print('type(x) = ', type(x))
print('x.shape = ', x.shape)
print('x.numel() = ', x.numel())
print('x.ndimension() = ', x.ndimension())
print('x.dtype = ', x.dtype)
print('x.device = ', x.device)

print('--------------------- 分割线 ---------------------')
print('# 1维向量')
x1 = tc.tensor([1])
print('x1 = ', x1)
print('x1.shape = ', x1.shape)
print('x1.numel() = ', x1.numel())
print('x1.ndimension() = ', x1.ndimension())

print('--------------------- 分割线 ---------------------')
print('# 向量')
y = tc.tensor([1, 3, 5, 7])
print('y = ', y)
print('y.shape = ', y.shape)
print('y.numel() = ', y.numel())
print('y.ndimension() = ', y.ndimension())

print('y.device = ', y.device)
if tc.cuda.is_available():
    print('CUDA available. Put x to CUDA...')
    y = y.to(device='cuda')
    print('y.device = ', y.device)

print('--------------------- 分割线 ---------------------')
print('# 向量切片')
print('y[2] = ', y[2])
print('y[-1] = ', y[-1])
print('y[1:3] = ', y[1:3])
print('y[1:] = ', y[1:])
print('y[:3] = ', y[:3])
print('y[:] = ', y[:])

print('--------------------- 分割线 ---------------------')
print('# 矩阵')
I = tc.tensor([[1, 0], [0, 1]])
print('I = \n', I)
print('I.ndimension() = ', I.ndimension())
print('I.shape = ', I.shape)

I1 = tc.eye(2)
print('I1 = \n', I1)

pauli_z = I1.clone()
pauli_z[1, 1] = -1
print('I1 = \n', I1)
print('pauli_z = \n', pauli_z)

K = I1 * 1
K[1, 1] = -1
print('I1 = \n', I1)
print('K = \n', K)

J = I
print('I = ', I)
J[1, 1] = -1
print('I = \n', I)
print('J = \n', J)

input()

print('--------------------- 分割线 ---------------------')
print('# 矩阵切片')
z = tc.randn((4, 4))
print('z = \n', z)
print('z[1, 1:3] = ', z[1, 1:3])
print('z[1, 1:] = ', z[1, 1:])
print('z[1, :3] = ', z[1, :3])
print('z[1, :] = ', z[1, :])
print('z[1:3, :2] = \n', z[1:3, :2])

print('--------------------- 分割线 ---------------------')
print('# delta张量')
delta = tc.tensor([[[1, 0], [0, 0]], [[0, 0], [0, 1]]])
print('delta = \n', delta)
print('delta.ndimension() = ', delta.ndimension())
print('delta.shape = ', delta.shape)

print('--------------------- 分割线 ---------------------')
print('# 随机张量')
w = tc.randn((2, 2, 2), device='cpu', dtype=tc.float64)
print('w = \n', w)

