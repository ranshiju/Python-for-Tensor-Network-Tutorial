import numpy as np
import copy
from scipy.sparse.linalg import eigsh


def ED_ground_state(hamilt, pos, v0=None, k=1):
    """
    每个局域哈密顿量的指标顺序满足: (bra0, bra1, ..., ket0, ket1, ...)
    例：求单个三角形上定义的反铁磁海森堡模型基态：
    H2 = hamiltonian_heisenberg('half', 1, 1, 1, 0, 0, 0, 0)
    e0, gs = ED_ground_state([H2.reshape(2, 2, 2, 2)]*3, [[0, 1], [1, 2], [0, 2]])
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