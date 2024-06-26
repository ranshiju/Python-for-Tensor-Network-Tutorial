import torch as tc
from Library.BasicFun import choose_device
from Library.QuantumState import TensorPureState
from Library.Hamiltonians import heisenberg
from Library.ED import ED_ground_state


delta = 1  # 各向异性参数
L = 4  # 自旋个数
pos = [[0, 1], [1, 2], [2, 3]]  # 相互作用位置
device = choose_device()
dtype = tc.float64

print('各向异性参数 = %g' % delta)
hamilt = heisenberg(jz=delta)
ge, gs = ED_ground_state(hamilt.reshape(2, 2, 2, 2), pos=pos)
print('\t 基态能量 = %g ' % ge)  # ED程序参考2.8节

# ====================================
print('设置虚时演化算法参数')
tau = 0.2  # 初始Trotter切片长度
time_it = 1000  # 演化次数
time_ob = 20  # 观测量计算间隔
e0_eps = 1e-3  # 基态能收敛判据
tau_min = 1e-5  # 终止循环tau判据

print('随机生成量子态...')
psi = TensorPureState(nq=L, device=device, dtype=tc.float64)

print('定义局域虚时演化算符')
hamilt = hamilt.to(device=device, dtype=dtype)
u = tc.matrix_exp(-tau * hamilt).reshape(2, 2, 2, 2)

print('开始虚时演化（tau = %g）' % tau)
e0_ = 1.0  # 暂存基态能
beta = 0.0  # 记录倒温度
for t in range(time_it):
    for p in pos:
        psi.act_single_gate(u, pos=p)
    psi.normalize()
    beta += tau
    if t % time_ob == 0:
        e0 = 0.0
        for p in pos:
            e0 += psi.observation(
                hamilt.reshape(2, 2, 2, 2), p)
        print('beta = %g, Eg = %g' % (beta, e0))
        if abs(e0 - e0_) < (e0_eps * tau):
            tau *= 0.5
            u = tc.matrix_exp(-tau * hamilt).reshape(
                2, 2, 2, 2)
            print('由于演化收敛，tau减小为%g' % tau)
        if tau < tau_min:
            print('tau = %g < %g, 演化终止...' % (
                tau, tau_min))
            break
        e0_ = e0

# 上述演化见函数Algorithms.PhysAlgoTensor.GS_ImagEvo_tensor
from Algorithms.PhysAlgoTensor import GS_ImagEvo_tensor
print('\n===================================')
print('调用GS_ImagEvo_tensor函数...')
para = {
        'length': L,  # 总自旋数
        'tau': 0.2,  # 初始Trotter切片长度
        'time_it': time_it,  # 演化次数
        'time_ob': time_ob,  # 观测量计算间隔
        'e0_eps': e0_eps,  # 基态能收敛判据
        'tau_min': tau_min,  # 终止循环tau判据
        'device': device,
        'dtype': dtype,
        'print': False
    }
e1 = GS_ImagEvo_tensor(hamilt.reshape(2, 2, 2, 2),
                       pos, para=para)[0]
print('得基态能 = %g' % e1)


