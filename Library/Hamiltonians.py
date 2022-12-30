import torch as tc
from Library.MathFun import pauli_operators


def heisenberg(jx=1, jy=1, jz=1, hx1=0, hy1=0, hz1=0, hx2=0, hy2=0, hz2=0):
    op = pauli_operators()
    hamilt = jx * tc.kron(op['x'], op['x']) + jy * tc.kron(op['y'], op['y']).real + jz * tc.kron(
        op['z'], op['z']) / 4
    if abs(hx1) > 1e-16 or abs(hx2) > 1e-16:
        hamilt += (tc.kron(op['id'], op['x']) * hx2 + tc.kron(op['x'], op['id']) * hx1) / 2
    if abs(hz1) > 1e-16 or abs(hz2) > 1e-16:
        hamilt += (tc.kron(op['id'], op['z']) * hz2 + tc.kron(op['z'], op['id']) * hz1) / 2
    if abs(hy1) > 1e-16 or abs(hy2) > 1e-16:
        hamilt = hamilt.to(dtype=tc.complex128) + (
                tc.kron(op['id'], op['y']) * hy2 + tc.kron(op['y'], op['id']) * hy1) / 2
    return hamilt


