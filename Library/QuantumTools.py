import torch as tc
import math
import random
import collections
import Library.BasicFun as bf
import Library.MathFun as mf
from Library.QuantumState import TensorPureState


def act_N_qubit_QFT(psi, pos=None):
    if type(psi) is not TensorPureState:
        psi = TensorPureState(psi)
        flagTPS = False
    else:
        flagTPS = True
    num_q = psi.tensor.ndimension()
    if pos is None:
        pos = list(range(num_q))
    for n in range(len(pos)):
        psi.act_single_gate(mf.hadamard(), [pos[n]])
        for n1 in range(n+1, len(pos)):
            psi.act_single_gate(mf.phase_shift(2*math.pi/(2**(n1+1))), [pos[n]], [pos[n1]])
    if flagTPS:
        return psi
    else:
        return psi.tensor


def qubit_state_sampling(state, num_sample=1000, counter=True):
    """
    :param state: quantum state
    :param num_sample: number of samples_v
    :param counter: whether counter
    :return: sampling results
    """
    p = state * state.conj()
    population = bf.binary_strings(state.numel())
    y = random.choices(population, weights=p.flatten(), k=num_sample)
    if counter:
        y = collections.Counter(y)
    return y


def reduced_density_matrix(psi, pos):
    ind = list(range(psi.ndimension()))
    dim = 1
    for n in pos:
        ind.remove(n)
        dim *= psi.shape[n]
    x = psi.permute(pos + ind).reshape(dim, -1)
    return x.mm(x.t().conj())


def vecs2product_state(vecs):
    if type(vecs) in [list, tuple]:
        psi = vecs[0]
        dims = [vecs[0].numel()]
        for n in range(1, len(vecs)):
            psi = psi.outer(vecs[n])
            dims.append(vecs[n].numel())
        return psi.reshape(dims)
    elif vecs.ndimension() == 2:
        # vecs的形状为：向量维数 * 向量（特征）个数
        psi = vecs[:, 0]
        for n in range(1, vecs.shape[1]):
            psi = psi.outer(vecs[:, n])
        return psi.reshape([vecs.shape[0]] * vecs.shape[1])
    else:
        # vecs.ndimension() == 3
        # vecs的形状为：样本数 * 向量维数 * 向量（特征）个数
        psi1 = list()
        for m in range(vecs.shape[0]):
            psi = vecs[m, :, 0]
            for n in range(1, vecs.shape[2]):
                psi = psi.outer(vecs[m, :, n]).flatten()
            psi1.append(psi.reshape([1] + ([vecs.shape[1]] * vecs.shape[2])))
        # retuen：样本数 * [直积态维数])
        return tc.cat(psi1, dim=0)



