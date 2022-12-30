import torch as tc
from Library.ADQC import ADGate


print('非参数门：哈德玛门h')
h = ADGate('h')
print('self.name =', h.name)
print('门的tensor = ')
print(h.tensor)

print('非参数门：翻转门（泡利算符）x')
sigma_x = ADGate('x')
print('门的tensor = ')
print(sigma_x.tensor)

print('参数门：旋转门rotate_x')
rx = ADGate('rotate_x', paras=0.5)
print('变分参数paras')
print(rx.paras)
print('量子门tensor')
print(rx.tensor)
print('更改参数后需运行renew_gate函数更新tensor')
rx.paras.data_adqc = tc.tensor(1.0)
print('renew_gate运行前，tensor为')
print(rx.tensor)
rx.renew_gate()
print('renew_gate运行后，tensor为')
print(rx.tensor)

print('隐门输入latent')
x = tc.randn(2, 2)
p, s, q = tc.linalg.svd(x)
print('奇异值分解所得的pq = \n', p.mm(q))
v = ADGate('latent', paras=x)
print('隐门tensor = \n', v.tensor)

print('直接输入量子门unitary')
u = ADGate('unitary', paras=q)
print('此时paras与tensor相等')
print(u.paras)
print(u.tensor)

