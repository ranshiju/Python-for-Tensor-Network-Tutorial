import torch as tc
import numpy as np
import copy
import Library.Hamiltonians as hm
from scipy.sparse.linalg import eigsh
from Library.QuantumState import TensorPureState
from Library.BasicFun import fprint


def ED_ground_state(hamilt, pos, v0=None, k=1):
    """
    !! 与Library.ED.ED_ground_state为同一函数
    每个局域哈密顿量的指标顺序满足: (bra0, bra1, ..., ket0, ket1, ...)
    例：求单个三角形上定义的反铁磁海森堡模型基态：
    H2 = hamiltonian_heisenberg('half', 1, 1, 1, 0, 0, 0, 0)
    e0, gs = ED_ground_state(
               [H2.reshape(2, 2, 2, 2)]*3,
               [[0, 1], [1, 2], [0, 2]])
    print(e0)

    :param hamilt: 局域哈密顿量
    :param pos: 每个局域哈密顿量作用的自旋
    :param v0: 初态
    :param k: 求解的低能本征态个数
    :return lm: 最大本征值
    :return v1: 最大本征向量
    """
    from scipy.sparse.linalg import LinearOperator as LinearOp

    def one_map_tensordot(v, hs, pos_hs, v_dims, ind_v):
        v = v.reshape(v_dims)
        _v = np.zeros(v.shape)
        for n, pos_now in enumerate(pos_hs):
            ind_contract = list()
            ind_new = copy.deepcopy(ind_v)
            for nn in range(len(pos_now)):
                ind_contract.append(ind_v.index(pos_now[nn]))
                ind_new.remove(pos_now[nn])
            ind_new += pos_now
            ind_permute = list(np.argsort(ind_new))
            _v = _v + np.tensordot(
                v, hs[n], [ind_contract, list(range(len(
                    pos_now)))]).transpose(ind_permute)
        return _v.reshape(-1, )

    # 自动获取总格点数
    n_site = 0
    for x in pos:
        n_site = max([n_site] + list(x))
    n_site += 1
    # 自动获取格点的维数
    if type(hamilt) not in (list, tuple):
        hamilt = [hamilt] * len(pos)
    d = hamilt[0].shape[0]
    dims = [d] * n_site
    dim_tot = np.prod(dims)
    # 初始化向量
    if v0 is None:
        v0 = eval('np.random.randn' + str(tuple(dims)))
        # v0 = np.random.randn(dims)
    else:
        v0 = v0.reshape(dims)
    v0 /= np.linalg.norm(v0)
    # 初始化指标顺序
    ind = list(range(n_site))
    h_effect = LinearOp((dim_tot, dim_tot), lambda vg: one_map_tensordot(
                        vg, hamilt, pos, dims, ind))
    lm, v1 = eigsh(h_effect, k=k, which='SA', v0=v0)
    return lm, v1


def ED_spin_chain(v0=None, k=1, para=None):
    para0 = {
        'length': 10,
        'jx': 1,
        'jy': 1,
        'jz': 1,
        'hx': 0,
        'hy': 0,
        'hz': 0,
        'bound_cond': 'open'  # open or periodic
    }
    if para is None:
        para = dict()
    para = dict(para0, **para)
    hamilts = hm.spin_chain_NN_hamilts(
        para['jx'], para['jy'], para['jz'], para['hx'],
        para['hy'], para['hz'], para['length'],
        2, para['bound_cond'])
    pos = hm.pos_chain_NN(
        para['length'], para['bound_cond'])
    lm, v = ED_ground_state(hamilts, pos, v0=v0, k=k)
    return lm, v, pos, para


def GS_ImagEvo_tensor(hamilt, pos, psi=None, para=None):
    from Algorithms import wheels_tebd as wh
    wh.check_hamilts_and_pos(hamilt, pos)
    para0 = {
        'length': 4,  # 总自旋数
        'tau': 0.1,  # 初始Trotter切片长度
        'time_it': 1000,  # 演化次数
        'time_ob': 20,  # 观测量计算间隔
        'e0_eps': 1e-3,  # 基态能收敛判据
        'tau_min': 1e-4,  # 终止循环tau判据
        'device': None,
        'dtype': tc.float64,
        'print': True,  # 是否打印
        'log_file': None  # 打印到文件
    }
    if para is None:
        para = dict()
    para = dict(para0, **para)

    if psi is None:
        psi = TensorPureState(
            nq=para['length'],
            device=para['device'], dtype=para['dtype'])
    else:
        psi = psi.to(device=para['device'],
                     dtype=para['dtype'])
        psi = TensorPureState(tensor=psi)

    if type(hamilt) is tc.Tensor:
        hamilt = hamilt.to(
            device=para['device'], dtype=para['dtype'])
    else:
        hamilt = [ham.to(
            device=para['device'], dtype=para['dtype'])
            for ham in hamilt]
    u = wh.hamilts2gates(hamilt, para['tau'])

    e0_ = 1.0  # 暂存基态能
    beta = 0.0  # 记录倒温度
    tau = para['tau']
    for t in range(para['time_it']):
        for p in range(len(pos)):
            if type(hamilt) is tc.Tensor:
                u1 = u.reshape(2, 2, 2, 2)
            else:
                u1 = u[p].reshape(2, 2, 2, 2)
            psi.act_single_gate(u1, pos=pos[p])
        psi.normalize()
        beta += tau
        if t % para['time_ob'] == 0:
            e0 = 0.0
            for p in pos:
                e0 += psi.observation(
                    hamilt.reshape(2, 2, 2, 2), p)
                fprint('beta = %g, Eg = %g' % (beta, e0),
                       print_screen=para['print'],
                       file=para['log_file'])
            if abs(e0 - e0_) < para['e0_eps'] * tau:
                tau *= 0.5
                u = wh.hamilts2gates(hamilt, tau)
                fprint('由于演化收敛，tau减小为%g' % tau,
                       print_screen=para['print'],
                       file=para['log_file'])
            if tau < para['tau_min']:
                fprint('tau = %g < %g, 演化终止...' % (
                    tau, para['tau_min']),
                       print_screen=para['print'],
                       file=para['log_file'])
                break
            e0_ = e0
    return e0_, psi.tensor, para

