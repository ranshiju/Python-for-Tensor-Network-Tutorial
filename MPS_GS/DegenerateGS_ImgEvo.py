import torch as tc
import numpy as np
from Library.BasicFun import choose_device
from Library.Hamiltonians import heisenberg
from Library.QuantumState import TensorPureState
from Algorithms.PhysAlgoTensor import \
    ED_ground_state, GS_ImagEvo_tensor


device = choose_device()
dtype = tc.float64
print('设置虚时演化算法参数')
para = {
        'tau': 0.2,  # 初始Trotter切片长度
        'time_it': 2000,  # 演化次数
        'time_ob': 20,  # 观测量计算间隔
        'e0_eps': 1e-4,  # 基态能收敛判据
        'tau_min': 1e-5,  # 终止循环tau判据
        'device': device,
        'dtype': dtype,
        'print': False
    }
rand_time = 20  # 随机初始化计算虚时演化次数
delta = 1  # 各向异性参数
hamilt = heisenberg(jz=delta)  # 二体哈密顿量

print('===============================')
print('基态非简并情况：偶数自旋个数')
para['length'] = 6  # 自旋个数
pos = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
ge, gs = ED_ground_state(
    hamilt.reshape(2, 2, 2, 2), pos=pos, k=6)
print('前6个低能激发能为 = ', ge)

gs = TensorPureState(tc.Tensor(gs[:, 0]).to(
    device=para['device'], dtype=para['dtype']
).reshape([2] * para['length']))
eb = gs.observe_bond_energies(hamilt, pos)
print('Bond基态能 = ')
print(eb.cpu().tolist())
ent = gs.bipartite_ent_entropy_all()
print('不同位置二分纠缠熵 = ')
print(ent.cpu().tolist())

psi = None
for t in range(rand_time):
    e1, psi1, _ = GS_ImagEvo_tensor(
        hamilt.reshape(2, 2, 2, 2), pos, para=para)
    if t > 0:
        fid = tc.abs(tc.inner(
            psi1.flatten(), psi.flatten())).item()
        print('对于第%i次计算，基态能为%g，'
              '基态与上次的保真度为%g' % (t, e1, fid))
    psi = psi1

print('==========================')
print('基态简并情况：奇数自旋个数')
para['length'] = 5  # 自旋个数
pos = [[0, 1], [1, 2], [2, 3], [3, 4]]  # 相互作用位置
ge, gs = ED_ground_state(
    hamilt.reshape(2, 2, 2, 2), pos=pos, k=6)
print('前6个低能激发能为 = ', ge)

psi = None
for t in range(rand_time):
    e1, psi1, _ = GS_ImagEvo_tensor(
        hamilt.reshape(2, 2, 2, 2), pos, para=para)
    if t > 0:
        fid = tc.abs(
            tc.inner(psi1.flatten(), psi.flatten()))
        print('对于第%i次计算，基态能为%g，'
              '基态与上次的保真度为%g' %
              (t, e1, fid.item()))
    psi = psi1
