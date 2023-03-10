import copy
import math

import torch as tc
import numpy as np


def entanglement_entropy(lm, eps=1e-15):
    lm1 = lm ** 2 + eps
    if type(lm1) is tc.Tensor:
        return tc.dot(-1 * lm1, tc.log(lm1))
    else:
        return np.inner(-1 * lm1, np.log(lm1))


def hadamard():
    return tc.tensor([[1.0, 1.0], [1.0, -1.0]]).to(dtype=tc.float64) / math.sqrt(2)


def hosvd(x, dc=None, return_lms=False):
    if type(dc) is int:
        dc = [dc] * x.ndimension()
    u, lms = list(), list()
    for n in range(x.ndimension()):
        mat = reduced_matrix(x, n)
        lm_, u_ = tc.linalg.eigh(mat)
        if (dc is not None) and (dc[n] < x.shape[n]):
            u_ = u_[:, -dc[n]:]
        u.append(u_)
        lms.append(lm_)
    core = tucker_product(x, u, conj=True, dim=0)
    if return_lms:
        return core, u, lms
    else:
        return core, u


def kron(mats):
    assert len(mats) >= 2
    if type(mats[0]) is tc.Tensor:
        mat = tc.kron(mats[0], mats[1])
        for n in range(2, len(mats)):
            mat = tc.kron(mat, mats[n])
    else:
        mat = np.kron(mats[0], mats[1])
        for n in range(2, len(mats)):
            mat = np.kron(mat, mats[n])
    return mat


def pauli_basis(which=None, device='cpu', if_list=False):
    basis = {
        'x': tc.tensor([[1.0, 1.0], [1.0, -1.0]], device=device) / np.sqrt(2),
        'y': tc.tensor([[1.0, 1.0j], [1.0, -1.0j]], device=device) / np.sqrt(2),
        'z': tc.eye(2, dtype=tc.float64, device=device)
    }
    if which is None:
        if if_list:
            return [basis['x'], basis['y'], basis['z']]
        else:
            return basis
    else:
        return basis[which]


def pauli_operators(which=None, device='cpu', if_list=False):
    op = {
        'x': tc.tensor([[0.0, 1.0], [1.0, 0.0]], device=device, dtype=tc.float64),
        'y': tc.tensor([[0.0, -1.0j], [1.0j, 0.0]], device=device, dtype=tc.complex128),
        'z': tc.tensor([[1.0, 0.0], [0.0, -1.0]], device=device, dtype=tc.float64),
        'id': tc.eye(2).to(device=device, dtype=tc.float64)
    }
    if which is None:
        if if_list:
            return [op['id'], op['x'], op['y'], op['z']]
        else:
            return op
    else:
        return op[which]


def phase_shift(theta):
    return tc.tensor([
        [1.0, 0.0],
        [0.0, 1j*math.exp(theta)]
    ]).to(dtype=tc.complex128)


def rank1_product(vecs, c=1.0):
    if type(vecs[0]) is np.ndarray:
        return rank1_product_np(vecs, c)
    else:
        return rank1_product_tc(vecs, c)


def rank1_product_np(vecs, c=1.0):
    x = vecs[0]
    dims = [vecs[0].size]
    for v in vecs[1:]:
        dims.append(v.size)
        # x = np.einsum('a,b->ab', x, v)
        x = np.outer(x, v)
        x = x.reshape(-1)
    return x.reshape(dims) * c


def rank1_product_tc(vecs, c=1.0):
    x = vecs[0]
    dims = [vecs[0].numel()]
    for v in vecs[1:]:
        dims.append(v.numel())
        # x = tc.einsum('a,b->ab', x, v)
        x = tc.outer(x, v)
        x = x.reshape(-1)
    return x.reshape(dims) * c


def rank1(x, v=None, it_time=2000, tol=1e-14):
    """
    :param x: tensor to be decomposed
    :param v: initial vectors (default: random)
    :param it_time: total iteration time
    :param tol: tolerance to break the iteration
    :return: vectors and factor of the rank-1 decomposition
    """
    if type(x) is np.ndarray:
        return rank1_np(x, v, it_time, tol)
    else:
        return rank1_tc(x, v, it_time, tol)


def rank1_np(x, v=None, it_time=10000, tol=1e-14):
    # 初始化向量组v
    if v is None:
        v = list()
        for n in range(x.ndim):
            v.append(np.random.randn(x.shape[n]))

    # 归一化向量组v
    for n in range(x.ndim):
        v[n] /= np.linalg.norm(v[n])

    norm1 = 1
    err = np.ones(x.ndim)
    err_norm = np.ones(x.ndim)
    for t in range(it_time):
        for n in range(x.ndim):
            x1 = copy.deepcopy(x)
            for m in range(n):
                x1 = np.tensordot(x1, v[m].conj(), [[0], [0]])
            for m in range(len(v)-1, n, -1):
                x1 = np.tensordot(x1, v[m].conj(), [[-1], [0]])
            norm = np.linalg.norm(x1)
            v1 = x1 / norm
            err[n] = np.linalg.norm(v[n] - v1)
            err_norm[n] = np.linalg.norm(norm - norm1)
            v[n] = v1
            norm1 = norm
        if err.sum()/x.ndim < tol and err_norm.sum()/x.ndim < tol:
            break

    return v, norm1


def rank1_tc(x, v=None, it_time=10000, tol=1e-14):
    # 初始化向量组v
    if v is None:
        v = list()
        for n in range(x.ndimension()):
            v.append(tc.randn(x.shape[n], device=x.device, dtype=x.dtype))

    # 归一化向量组v
    for n in range(x.ndimension()):
        v[n] /= v[n].norm()

    norm1 = 1
    err = tc.ones(x.ndimension(), device=x.device, dtype=tc.float64)
    err_norm = tc.ones(x.ndimension(), device=x.device, dtype=tc.float64)
    for t in range(it_time):
        for n in range(x.ndimension()):
            x1 = x.clone()
            for m in range(n):
                x1 = tc.tensordot(x1, v[m].conj(), [[0], [0]])
            for m in range(len(v)-1, n, -1):
                x1 = tc.tensordot(x1, v[m].conj(), [[-1], [0]])
            norm = x1.norm()
            v1 = x1 / norm
            err[n] = (v[n] - v1).norm()
            err_norm[n] = (norm - norm1).norm()
            v[n] = v1
            norm1 = norm
        if err.sum()/x.ndimension() < tol and err_norm.sum()/x.ndimension() < tol:
            break

    return v, norm1


def reduced_matrix(x, bond):
    indexes = list(range(x.ndimension()))
    indexes.pop(bond)
    s = x.shape
    x1 = x.permute([bond] + indexes).reshape(s[bond], -1)
    return x1.mm(x1.t().conj())


def rotate(paras):
    alpha, beta, delta, theta = paras[0], paras[1], paras[2], paras[3]
    gate = tc.ones((2, 2), device=paras.device, dtype=tc.complex128)
    gate[0, 0] = tc.exp(1j * (delta - alpha / 2 - beta / 2)) * tc.cos(theta / 2)
    gate[0, 1] = -tc.exp(1j * (delta - alpha / 2 + beta / 2)) * tc.sin(theta / 2)
    gate[1, 0] = tc.exp(1j * (delta + alpha / 2 - beta / 2)) * tc.sin(theta / 2)
    gate[1, 1] = tc.exp(1j * (delta + alpha / 2 + beta / 2)) * tc.cos(theta / 2)
    return gate


def rotate_pauli(theta, direction):
    op = pauli_operators(direction).to(theta.device)
    return tc.matrix_exp(-1j * theta / 2 * op)


def series_sin_cos(x, coeff_sin, coeff_cos, k_step=0.02):
    y = tc.ones(x.numel(), device=x.device, dtype=x.dtype) * coeff_cos[0]
    coeff_sin, coeff_cos = coeff_sin.to(device=x.device, dtype=x.dtype), coeff_cos.to(device=x.device, dtype=x.dtype)
    for n in range(1, len(coeff_sin)):
        y = y + tc.sin(x * coeff_sin[n] * n * k_step)
    for n in range(1, len(coeff_cos)):
        y = y + tc.cos(coeff_cos[n] * n * k_step)
    return y


def sign_accuracy(pred, labels):
    pred = pred * labels
    return tc.sum(pred > 0) / labels.numel()


def spin_operators(spin, is_torch=True):
    op = dict()
    if spin.lower() == 'half':
        op_ = pauli_operators()
        op['id'] = tc.eye(2)
        op['sx'] = op_['x'] / 2
        op['sy'] = op_['y'] / 2
        op['sz'] = op_['z'] / 2
        op['su'] = tc.zeros((2, 2))
        op['sd'] = tc.zeros((2, 2))
        op['su'][0, 1] = 1
        op['sd'][1, 0] = 1
    elif spin.lower() in ['1', 'one']:
        op['id'] = tc.eye(3)
        op['sx'] = tc.zeros((3, 3))
        op['sy'] = tc.zeros((3, 3), dtype=tc.complex128)
        op['sz'] = tc.zeros((3, 3))
        op['sx'][0, 1] = 1
        op['sx'][1, 0] = 1
        op['sx'][1, 2] = 1
        op['sx'][2, 1] = 1
        op['sy'][0, 1] = -1j
        op['sy'][1, 0] = 1j
        op['sy'][1, 2] = -1j
        op['sy'][2, 1] = 1j
        op['sz'][0, 0] = 1
        op['sz'][2, 2] = -1
        op['sx'] /= 2 ** 0.5
        op['sy'] /= 2 ** 0.5
        op['su'] = tc.real(op['sx'] + 1j * op['sy'])
        op['sd'] = tc.real(op['sx'] - 1j * op['sy'])
    if not is_torch:
        for k in op:
            op[k] = op[k].numpy()
    return op


def super_diagonal_tensor(dim, order):
    """
    :param dim: bond dimension
    :param order: tensor order
    :return: high-order super-orthogonal tensor
    """
    delta = tc.zeros([dim] * order, dtype=tc.float64)
    for n in range(dim):
        x = (''.join([str(n), ','] * order))
        exec('delta[' + x[:-1] + '] = 1.0')
    return delta


def swap():
    return tc.eye(4).reshape(2, 2, 2, 2).permute(1, 0, 2, 3).to(dtype=tc.float64)
    # return tc.einsum('ab,ij->aijb', tc.eye(2), tc.eye(2)).to(dtype=tc.float64)


def ttd_np(x, chi=-1, svd=True):
    """
    :param x: tensor to be decomposed
    :param chi: dimension cut-off. Use QR decomposition when chi=None;
                use SVD but don'tc truncate when chi=-1
    :return tensors: tensors in the TT form
    :return lm: singular values in each decomposition (calculated when chi is not None)
    """
    dims = x.shape
    ndim = x.ndim
    dimL = 1
    tensors = list()
    lm = list()
    if chi is None:
        chi = -1
    for n in range(ndim-1):
        if (chi < 0) and (not svd):  # No truncation
            q, x = np.linalg.qr(x.reshape(dimL*dims[n], -1))
            dimL1 = x.shape[0]
        else:
            q, s, v = np.linalg.svd(x.reshape(dimL*dims[n], -1))
            if chi > 0:
                dc = min(chi, s.size)
            else:
                dc = s.size
            q = q[:, :dc]
            s = s[:dc]
            lm.append(s)
            x = np.diag(s).dot(v[:dc, :])
            dimL1 = dc
        tensors.append(q.reshape(dimL, dims[n], dimL1))
        dimL = dimL1
    tensors.append(x.reshape(dimL, dims[-1]))
    tensors[0] = tensors[0][0, :, :]
    return tensors, lm


def ttd_tc(x, chi=-1, svd=True):
    """
    :param x: tensor to be decomposed
    :param chi: dimension cut-off. Use QR decomposition when chi=None;
                use SVD but don'tc truncate when chi=-1
    :param svd: use svd or qr
    :return tensors: tensors in the TT form
    :return lm: singular values in each decomposition (calculated when chi is not None)
    """
    dims = x.shape
    ndim = x.ndimension()
    dimL = 1
    tensors = list()
    lm = list()
    if chi is None:
        chi = -1
    for n in range(ndim-1):
        if (chi < 0) and (not svd):  # No truncation
            q, x = tc.linalg.qr(x.reshape(dimL*dims[n], -1))
            dimL1 = x.shape[0]
        else:
            q, s, v = tc.linalg.svd(x.reshape(dimL*dims[n], -1))
            if chi > 0:
                dc = min(chi, s.numel())
            else:
                dc = s.numel()
            q = q[:, :dc]
            s = s[:dc]
            lm.append(s)
            x = tc.diag(s).mm(v[:dc, :])
            dimL1 = dc
        tensors.append(q.reshape(dimL, dims[n], dimL1))
        dimL = dimL1
    tensors.append(x.reshape(dimL, dims[-1]))
    tensors[0] = tensors[0][0, :, :]
    return tensors, lm


def ttd(x, chi=None):
    """
    :param x: tensor to be decomposed
    :param chi: dimension cut-off. Use QR decomposition when chi=None;
                use SVD but don'tc truncate when chi=-1
    :return tensors: tensors in the TT form
    :return lm: singular values in each decomposition (calculated when chi is not None)
    """
    if type(x) is np.ndarray:
        return ttd_np(x, chi)
    else:
        return ttd_tc(x, chi)


def tucker_product(tensor, mats, pos=None, dim=1, conj=False):
    """
    :param tensor: 张量
    :param mats: 变换矩阵
    :param pos: 变换矩阵与张量的第几个指标收缩
    :param dim: 收缩各个变换矩阵的第几个指标
    :return G: 返回Tucker乘积的结果
    """
    from Library.BasicFun import inverse_permutation
    if pos is None:
        assert len(mats) == tensor.ndimension()
        pos = list(range(len(mats)))
    ind = list(range(tensor.ndimension()))
    for n in range(len(pos)):
        pos_now = ind.index(pos[n])
        if conj:
            tensor = tc.tensordot(tensor, mats[n].conj(), [[pos_now], [dim]])
        else:
            tensor = tc.tensordot(tensor, mats[n], [[pos_now], [dim]])
        p = ind.pop(pos_now)
        ind += [p]
    order = inverse_permutation(ind)
    return tensor.permute(order)


def tucker_rank(x, eps=1e-14):
    lms = hosvd(x, return_lms=True)[2]
    r = list()
    for lm in lms:
        r_ = (lm > eps).sum().item()
        r.append(r_)
    return r



