import torch as tc
from Library.MathFun import pauli_operators


def heisenberg(jx=1, jy=1, jz=1, hx1=0, hy1=0, hz1=0, hx2=0, hy2=0, hz2=0):
    op = pauli_operators()
    hamilt = (jx * tc.kron(op['x'], op['x']) + jy * tc.kron(op['y'], op['y']).real + jz * tc.kron(
        op['z'], op['z'])) / 4
    if abs(hx1) > 1e-16 or abs(hx2) > 1e-16:
        hamilt += (tc.kron(op['id'], op['x']) * hx2 + tc.kron(op['x'], op['id']) * hx1) / 2
    if abs(hz1) > 1e-16 or abs(hz2) > 1e-16:
        hamilt += (tc.kron(op['id'], op['z']) * hz2 + tc.kron(op['z'], op['id']) * hz1) / 2
    if abs(hy1) > 1e-16 or abs(hy2) > 1e-16:
        hamilt = hamilt.to(dtype=tc.complex128) + (
                tc.kron(op['id'], op['y']) * hy2 + tc.kron(op['y'], op['id']) * hy1) / 2
    return hamilt


def spin_chain_NN_hamilts(jx, jy, jz, hx, hy, hz, length, d, bound_cond):
    hamilt = heisenberg(jx, jy, jz, hx/2, hy/2, hz/2,
                        hx/2, hy/2, hz/2).reshape([d] * 4)
    bound_cond = bound_cond.lower()
    assert bound_cond in ['open', 'periodic']
    if bound_cond == 'periodic':
        hamilts = [hamilt] * length
    else:
        assert length > 2
        hamiltL = heisenberg(jx, jy, jz, hx, hy, hz,
                             hx/2, hy/2, hz/2).reshape([d] * 4)
        hamiltR = heisenberg(jx, jy, jz, hx/2, hy/2, hz/2,
                             hx, hy, hz).reshape([d] * 4)
        hamilts = [hamiltL] + [hamilt] * (length - 3) + [hamiltR]
    return hamilts


def pos_chain_NN(length, bound_cond='open'):
    pos = [[x, x + 1] for x in range(length - 1)]
    bound_cond = bound_cond.lower()
    assert bound_cond in ['open', 'periodic', 'obc', 'pbc']
    if bound_cond in ['periodic', 'pbc']:
        pos = pos + [[0, length - 1]]
    return pos


def pos_square_NN(size, bound_cond='open'):
    pos = list()
    width, height = tuple(size)
    for i in range(0, width - 1):  # interactions in the first row
        pos.append([i, i + 1])
    for n in range(1, height):  # interactions in the n-th row
        for i in range(0, width - 1):
            pos.append([n * width + i, n * width + i + 1])
    for n in range(0, width):
        for i in range(0, height - 1):
            pos.append([i * width + n, (i + 1) * width + n])
    if bound_cond.lower() in ['periodic', 'pbc']:
        for n in range(0, height):
            pos.append([n * width, (n + 1) * width - 1])
        for n in range(0, width):
            pos.append([n, (height - 1) * width + n])
    return pos

