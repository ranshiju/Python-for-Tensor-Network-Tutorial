import torch as tc
import matplotlib.pylab as plt
from torch.optim import Adam
from Library import ADQC
from Library.QuantumState import state_all_up
from Library.BasicFun import choose_device


lr, it_time = 1e-3, 5000  # 初始学习率与迭代总次数
device = choose_device()

print('随机生成目标态')
num_qubits = 3
psi_target = tc.randn((2**num_qubits, ),
                      dtype=tc.complex128, device=device)
psi_target /= psi_target.norm()

print('建立ADQC_basic实例')
qc = ADQC.ADQC_basic()
print('在ADQC_basic实例中添加ADgate实例')
qc.add_ADgates(ADQC.ADGate('rotate', pos=0))
qc.add_ADgates(ADQC.ADGate('x', pos=1, pos_control=0))
qc.add_ADgates(ADQC.ADGate('rotate', pos=0))
qc.add_ADgates(ADQC.ADGate('rotate', pos=1))
qc.add_ADgates(ADQC.ADGate('x', pos=2, pos_control=1))
qc.add_ADgates(ADQC.ADGate('rotate', pos=1))
qc.add_ADgates(ADQC.ADGate('rotate', pos=2))
qc.add_ADgates(ADQC.ADGate('x', pos=0, pos_control=2))
qc.add_ADgates(ADQC.ADGate('rotate', pos=2))

print('打印各个门及变分参数')
for x in qc.state_dict():
    print(x, qc.state_dict()[x])

optimizer = Adam(qc.parameters(), lr=lr)  # 建立优化器
loss_rec = tc.zeros(it_time, )
print('开始优化')
for t in range(it_time):
    # 建立初态|000>
    psi = state_all_up(n_qubit=3, d=2).to(
        device=device, dtype=psi_target.dtype)
    psi = qc(psi)
    loss = 1 - psi.flatten().dot(psi_target.conj()).norm()
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
