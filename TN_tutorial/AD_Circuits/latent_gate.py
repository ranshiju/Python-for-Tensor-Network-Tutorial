import torch as tc
import matplotlib.pylab as plt
import Library.QuantumState as qs
from torch.optim import Adam


lr, it_time = 1e-3, 3000  # 初始学习率与迭代总次数

print('随机生成目标态')
num_qubits, num_gates = 3, 4
psi_target = tc.randn((2**num_qubits, ), dtype=tc.complex128)
psi_target /= psi_target.norm()

print('初始化4个隐门并建立优化器')
V = [tc.randn((4, 4), dtype=tc.complex128,
              requires_grad=True) for _ in range(num_gates)]
optimizer = Adam(V, lr=lr)

loss_rec = tc.zeros(it_time, )
print('开始优化')
for t in range(it_time):
    # 从隐门生成幺正门
    U = list()
    for x in V:
        P, S, Q = tc.linalg.svd(x)
        U.append(P.mm(Q))
    # 建立初态|000>
    psi = qs.state_all_up(n_qubit=3, d=2)
    psi = qs.TensorPureState(psi)
    # 作用各个量子门
    psi.act_single_gate(U[0], [0, 1])
    psi.act_single_gate(U[1], [1, 2])
    psi.act_single_gate(U[2], [0, 1])
    psi.act_single_gate(U[3], [1, 2])

    loss = 1 - psi.tensor.conj().flatten().dot(
        psi_target).norm()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    loss_rec[t] = loss.item()
    if t % 20 == 19:
        print('第%d次迭代，loss = %g' % (t, loss.item()))

plt.plot(loss_rec)
plt.xlabel('iteration time')
plt.ylabel('loss')
plt.show()





