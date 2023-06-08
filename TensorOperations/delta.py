import torch as tc


def super_diagonal_tensor(dim, order):
    """
    注：该函数已收录于MathFun模块
    :param dim: bond dimension
    :param order: tensor order_g
    :return: high-order_g super-orthogonal tensor
    """
    delta = tc.zeros([dim] * order, dtype=tc.float64)
    for n in range(dim):
        x = (''.join([str(n), ','] * order))
        exec('delta[' + x[:-1] + '] = 1.0')
    return delta


U = tc.randn((2, 2), dtype=tc.float64)
V = tc.randn((2, 2), dtype=tc.float64)
W = tc.randn((2, 2), dtype=tc.float64)
gamma = tc.randn(2, dtype=tc.float64)

T1 = tc.einsum('n,an,bn,cn->abc', gamma, U, V, W)

d = super_diagonal_tensor(2, 4)
T2 = tc.einsum('i,aj,bk,cl,ijkl->abc', gamma, U, V, W, d)

print('|T1 - T2| = ', tc.norm(T1 - T2))


