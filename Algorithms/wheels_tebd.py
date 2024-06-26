import torch as tc


def check_hamilts_and_pos(hamilts, pos):
    for p in pos:
        assert p[0] < p[1]
    if type(hamilts) is tc.Tensor:
        for p in pos:
            assert len(p) * 2 == hamilts.ndimension()
    else:
        for p in range(len(pos)):
            assert len(pos[p]) * 2 == hamilts[p].ndimension()


def find_optimal_new_center(l0, r0, l1, r1):
    l_min = min([abs(l0-l1), abs(l0-r1)])
    r_min = min([abs(r0-l1), abs(r0-r1)])
    if l_min < r_min:
        return 'rl'
    else:
        return 'lr'  # r0离新区域近


def hamilts2gates(hamilts, tau):
    # hamilts中的哈密顿量需要时2N阶张量，N为自旋个数
    assert type(hamilts) in [list, tuple, tc.Tensor]
    if type(hamilts) is tc.Tensor:
        d = hamilts.shape[0]
        gates = tc.matrix_exp(-tau * hamilts.reshape(d ** (round(hamilts.ndimension() / 2)), -1))
    else:
        d = hamilts[0].shape[0]
        gates = [tc.matrix_exp(-tau * g.reshape(d ** (
            round(g.ndimension() / 2)), -1)) for g in hamilts]
    return gates


