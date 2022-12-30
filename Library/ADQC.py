import math

import torch as tc
from torch import nn

from Library import MathFun as mf
from Library import BasicFun as bf
from Library.QuantumState import state_all_up


class ADGate(nn.Module):

    def __init__(self, name, pos=None, pos_control=None, paras=None, requires_grad=True,
                 qudit_dims=None, settings=None, device=None, dtype=tc.complex128):
        """
        :param name: which tensor
        :param pos: positions of the qubits
        :param pos_control: positions of the controlling qubits
        :param paras: variational parameters (if any)
        :param requires_grad: whether requiring grad
        :param qudit_dims: dimensions of the qudits (needed only d>2 qudits exist)
        :param settings: a dict of other settings
                {'initial_way': the way of initialize the tensor for latent gates}
        :param device: device
        :param dtype: dtype
        """
        super(ADGate, self).__init__()
        self.device = bf.choose_device(device)
        self.dtype = dtype
        self.pos = pos
        self.pos_control = pos_control
        self.requires_grad = requires_grad
        self.name = name.lower()
        self.settings = settings
        self.spin_op = None
        self.paras = paras
        self.tensor = None

        self.preprocess()
        self.variational = self.requires_grad

        if self.name in ['x', 'y', 'z']:
            self.tensor = mf.pauli_operators(self.name)
            self.tensor = self.tensor.to(device=self.device, dtype=self.dtype)
            self.variational = False
        elif self.name in ['hadamard', 'h']:
            self.tensor = mf.hadamard()
            self.tensor = self.tensor.to(device=self.device, dtype=self.dtype)
            self.variational = False
# ==================== 以上是非参数门，以下是参数门 ====================
        elif self.name == 'rotate':
            if paras is None:
                self.paras = tc.randn((4, ))
            self.paras = self.paras.to(device=self.device)
        elif self.name in ['rotate_x', 'rotate_y', 'rotate_z', 'phase_shift']:
            if paras is None:
                self.paras = tc.randn(1)
            self.paras = self.paras.to(device=self.device)
        elif self.name == 'evolve_variational_mag':  # 单体磁场演化, shape=(3, )
            assert 'tau' in self.settings
            if paras is None:
                self.paras = tc.randn((3, ))
            self.paras = self.paras.to(device=self.device, dtype=self.dtype)
            if self.paras[1].abs() > 1e-12:
                assert self.dtype in [tc.complex128, tc.complex64, tc.complex32]
        elif self.name == 'latent':
            if paras is None:
                if self.pos is None:
                    ndim = 2
                else:
                    ndim = len(self.pos)
                if qudit_dims is None:
                    qudit_dims = [2] * ndim
                dim_t = math.prod(qudit_dims)
                if 'initial_way_latent' in self.settings:
                    if self.settings['initial_way_latent'] == 'identity':
                        self.paras = tc.eye(dim_t, dim_t) + 1e-5 * tc.randn((dim_t, dim_t))
                else:
                    self.paras = tc.randn((dim_t, dim_t))
            self.paras = self.paras.to(device=self.device, dtype=self.dtype)
        elif self.name == 'arbitrary':  # 无限制门（paras即门）
            assert type(paras) is tc.Tensor
            self.paras = paras.to(device=self.device, dtype=self.dtype)
            self.tensor = self.paras
        if self.variational:
            self.paras = nn.Parameter(self.paras, requires_grad=True)
        self.renew_gate()

    def preprocess(self):
        if self.name == 'not':
            self.name = 'x'
        if (self.paras is not None) and (type(self.paras) is not tc.Tensor):
            self.paras = tc.tensor(self.paras)
        if self.pos_control is None:
            self.pos_control = list()
        if self.settings is None:
            self.settings = dict()
        if type(self.pos) is int:
            self.pos = [self.pos]
        if type(self.pos_control) is int:
            self.pos_control = [self.pos_control]

    def renew_gate(self):
        if self.name == 'phase_shift':
            self.tensor = mf.phase_shift(self.paras)
        elif self.name == 'rotate':
            self.tensor = mf.rotate(self.paras)
        elif self.name in ['rotate_x', 'rotate_y', 'rotate_z']:
            direction = self.name[-1]
            self.tensor = mf.rotate_pauli(self.paras, direction)
        elif self.name == 'evolve_variational_mag':
            if self.spin_op is None:
                self.spin_op = mf.spin_operators('half')
                for x in self.spin_op:
                    self.spin_op[x] = self.spin_op[x].to(device=self.device, dtype=self.dtype)
            h = self.paras[0] * self.spin_op['sx'] + self.paras[1] * self.spin_op['sy'] + \
                self.paras[2] * self.spin_op['sz']
            h = h.to(device=self.device, dtype=self.dtype)
            self.tensor = tc.matrix_exp(-1j * self.settings['tau'] * h)
        elif self.name == 'latent':
            self.tensor = self.latent2unitary(self.paras)
        elif self.name == 'arbitrary':
            self.tensor = self.paras

    @staticmethod
    def latent2unitary(g):
        u, _, v = tc.linalg.svd(g)
        return u.mm(v)


class ADQC_basic(nn.Module):

    def __init__(self, device=None, dtype=tc.complex128):
        super(ADQC_basic, self).__init__()
        self.single_state = True
        self.device = bf.choose_device(device)
        self.dtype = dtype
        self.layers = nn.Sequential()

    def act_nth_gate(self, state, n):
        m_p = len(self.layers[n].pos)
        m_c = len(self.layers[n].pos_control)
        n_qubit = state.ndimension()
        shape = state.shape
        perm = list(range(n_qubit))
        for pp in self.layers[n].pos:
            perm.remove(pp)
        for pp in self.layers[n].pos_control:
            perm.remove(pp)
        perm = self.layers[n].pos + perm + self.layers[n].pos_control
        state1 = state.permute(perm).reshape(2 ** m_p, -1, 2 ** m_c)
        state1_ = state1[:, :, :-1]
        state2_ = self.layers[n].tensor.reshape(-1, 2 ** m_p).mm(state1[:, :, -1])
        state1 = tc.cat([state1_, state2_.reshape(state2_.shape + (1,))], dim=-1)
        state1 = state1.reshape(shape)
        return state1.permute(bf.inverse_permutation(perm))

    def act_nth_gate_multi_states(self, states, n):
        m_p = len(self.layers[n].pos)
        m_c = len(self.layers[n].pos_control)
        n_qubit = states.ndimension() - 1
        shape = states.shape
        states1 = states.permute(list(range(1, n_qubit+1)) + [0])
        perm = list(range(n_qubit))
        for pp in self.layers[n].pos:
            perm.remove(pp)
        for pp in self.layers[n].pos_control:
            perm.remove(pp)
        perm = self.layers[n].pos + perm + self.layers[n].pos_control
        states1 = states1.permute(perm + [n_qubit]).reshape(2 ** m_p, -1, 2 ** m_c, shape[0])
        state1_ = states1[:, :, :-1, :]
        state2_ = tc.einsum('ab,bcn->acn', self.layers[n].tensor.reshape(-1, 2 ** m_p), states1[:, :, -1, :])
        # state2_ = self.layers[n].tensor.reshape(-1, 2 ** m_p).mm(states1[:, :, -1, :])
        s_ = state2_.shape
        state2_ = state2_.reshape(s_[0], s_[1], 1, s_[2])
        states1 = tc.cat([state1_, state2_], dim=2)
        states1 = states1.reshape(shape[1:] + (shape[0],))
        perm1 = [m+1 for m in perm] + [0]
        return states1.permute(bf.inverse_permutation(perm1))

    @staticmethod
    def act_single_gate(state, gate, pos, pos_control):
        m_p = len(pos)
        m_c = len(pos_control)
        n_qubit = state.ndimension()
        shape = state.shape
        perm = list(range(n_qubit))
        for pp in pos:
            perm.remove(pp)
        for pp in pos_control:
            perm.remove(pp)
        perm = pos + perm + pos_control
        state1 = state.permute(perm).reshape(2 ** m_p, -1, 2 ** m_c)
        state1_ = state1[:, :, :-1]
        state2_ = gate.reshape(-1, 2 ** m_p).mm(state1[:, :, -1])
        state1 = tc.cat([state1_, state2_.reshape(state2_.shape + (1,))], dim=-1)
        state1 = state1.reshape(shape)
        return state1.permute(bf.inverse_permutation(perm))

    @staticmethod
    def act_single_ADgate(state, gate):
        m_p = len(gate.pos)
        m_c = len(gate.pos_control)
        n_qubit = state.ndimension()
        shape = state.shape
        perm = list(range(n_qubit))
        for pp in gate.pos:
            perm.remove(pp)
        for pp in gate.pos_control:
            perm.remove(pp)
        perm = gate.pos + perm + gate.pos_control
        state1 = state.permute(perm).reshape(2 ** m_p, -1, 2 ** m_c)
        state1_ = state1[:, :, :-1]
        state2_ = gate.tensor.reshape(-1, 2 ** m_p).mm(state1[:, :, -1])
        state1 = tc.cat([state1_, state2_.reshape(state2_.shape + (1,))], dim=-1)
        state1 = state1.reshape(shape)
        return state1.permute(bf.inverse_permutation(perm))

    def add_ADgates(self, gates, name=None):
        if type(gates) is ADGate:
            gates = [gates]
        for x in gates:
            if name is None:
                name = str(len(self.layers)) + '_' + x.name
            self.layers.add_module(name, x)

    def forward(self, state):
        self.renew_gates()
        if self.single_state:
            for n in range(len(self.layers)):
                state = self.act_nth_gate(state, n)
        else:
            for n in range(len(self.layers)):
                state = self.act_nth_gate_multi_states(state, n)
        return state

    def renew_gates(self):
        for n in range(len(self.layers)):
            self.layers[n].renew_gate()


class ADQC_LatentGates(ADQC_basic):

    def __init__(self, pos_one_layer=None, lattice='brick',
                 num_q=10, depth=3, ini_way='random',
                 device=None, dtype=tc.complex128):
        super(ADQC_LatentGates, self).__init__(
            device=device, dtype=dtype)
        self.lattice = lattice.lower()
        self.depth = depth
        self.ini_way = ini_way
        if pos_one_layer is None:
            self.pos = position_one_layer(self.lattice, num_q)
        else:
            self.pos = pos_one_layer

        paras = None
        for nd in range(depth):
            for ng in range(len(self.pos)):
                if self.ini_way == 'identity':
                    paras = tc.randn((4, 4)) * 1e-8 + tc.eye(4, 4)
                    paras = paras.to(device=self.device, dtype=self.dtype)
                name = self.lattice + '_layer' + str(nd) + '_gate' + str(ng)
                gate = ADGate(
                    'latent', pos=self.pos[ng], paras=paras,
                    device=self.device, dtype=self.dtype)
                self.layers.add_module(name, gate)


class QRNN_LatentGates(ADQC_basic):

    def __init__(self, pos_one_layer=None, lattice='brick', ini_way='random',
                 num_ancillary=6, depth=3, unitary=True, device=None,
                 dtype=tc.complex128):
        super(QRNN_LatentGates, self).__init__(
            device=device, dtype=dtype)
        self.lattice = lattice
        self.depth = depth
        self.num_a = num_ancillary
        self.ini_way = ini_way
        self.unitary = unitary
        if pos_one_layer is None:
            self.pos = position_one_layer(self.lattice, self.num_a+1)
        else:
            self.pos = pos_one_layer

        paras = None
        for nd in range(depth):
            for ng in range(len(self.pos)):
                if self.ini_way == 'identity':
                    paras = tc.randn((4, 4)) * 1e-5 + tc.eye(4, 4)
                    paras = paras.to(device=self.device, dtype=self.dtype)
                name = 'layer' + str(nd) + '_gate' + str(ng)
                if self.unitary:
                    gate = ADGate(
                        'latent', pos=self.pos[ng], paras=paras,
                        device=self.device, dtype=self.dtype)
                else:
                    gate = ADGate(
                        'arbitrary', pos=self.pos[ng], paras=paras,
                        device=self.device, dtype=self.dtype)
                self.layers.add_module(name, gate)

    def forward(self, vecs, psi=None, eps=1e-12):
        # vecs的形状为(样本数，向量维数，向量个数)
        if psi is None:
            psi = tc.zeros(2 ** self.num_a, device=self.device, dtype=self.dtype)
            psi[0] = 1.0
            psi = psi.repeat(vecs.shape[0], 1)
        self.renew_gates()
        norm = None
        # psi1 = psi * 1.0
        for n in range(vecs.shape[2]):
            # psi = psi + psi1
            # norm = tc.einsum('na,na->n', psi, psi.conj())
            # psi = tc.einsum('na,n->na', psi, 1 / (tc.sqrt(norm + eps)))
            psi = tc.einsum('na,nb->nab', psi, vecs[:, :, n])
            psi = psi.reshape([psi.shape[0]] + [2] * self.num_a + [vecs.shape[1]])
            for m in range(len(self.layers)):
                psi = self.act_nth_gate_multi_states(psi, m)
            psi = psi.reshape(-1, vecs.shape[1])[:, 0].reshape(vecs.shape[0], -1)
            norm = tc.einsum('na,na->n', psi, psi.conj())
            psi = tc.einsum('na,n->na', psi, 1/(tc.sqrt(norm+eps)))
            # psi1 = psi * 1.0
        return norm


def act_single_ADgate(state, gate):
    assert type(gate) is ADGate
    if type(state) is tc.Tensor:
        m_p = len(gate.pos)
        m_c = len(gate.pos_control)
        n_qubit = state.ndimension()
        shape = state.shape
        perm = list(range(n_qubit))
        for pp in gate.pos:
            perm.remove(pp)
        for pp in gate.pos_control:
            perm.remove(pp)
        perm = gate.pos + perm + gate.pos_control
        state1 = state.permute(perm).reshape(2 ** m_p, -1, 2 ** m_c)
        state1_ = state1[:, :, :-1]
        state2_ = gate.tensor.reshape(-1, 2 ** m_p).mm(state1[:, :, -1])
        state1 = tc.cat([state1_, state2_.reshape(state2_.shape + (1,))], dim=-1)
        state1 = state1.reshape(shape)
        return state1.permute(bf.inverse_permutation(perm))
    else:  # state为TensorPureState类
        state.act_single_gate(gate.tensor, gate.pos, gate.pos_control)


def get_diff_tensor(g, pos_diff):
    return ADGate('unitary', g.pos, [pos_diff] + g.pos_control, paras=g.diff_gate().tensor)


def position_one_layer(pattern, num_q):
    pos = list()
    if pattern == 'stair':
        for m in range(num_q-1):
            pos.append([m, m+1])
    else:  # brick
        m = 0
        while m < num_q-1:
            pos.append([m, m+1])
            m += 2
        m = 1
        while m < num_q-1:
            pos.append([m, m+1])
            m += 2
    return pos