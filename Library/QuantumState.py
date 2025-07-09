import sys
import random
from collections import Counter
import torch as tc
import numpy as np
import Library.BasicFun as bf
import Library.MathFun as mf


def state_all_up(n_qubit, d):
    x = tc.zeros((d ** n_qubit, ))
    x[0] = 1.0
    return x.reshape([d] * n_qubit)


def state_ghz(n_qubit):
    return mf.super_diagonal_tensor(2, n_qubit) / np.sqrt(2)


class TensorPureState:

    def __init__(self, tensor=None, nq=None,
                 device=None, dtype=None):
        self.rand_seed = None
        if (device is None) and (type(tensor) is tc.Tensor):
            device = tensor.device
        self.device = bf.choose_device(device)
        if (dtype is None) and (type(tensor) is tc.Tensor):
            dtype = tensor.dtype
        self.dtype = dtype
        if nq is None:
            nq = 3
        if tensor is None:
            self.tensor = tc.randn([2] * nq, device=self.device,
                                   dtype=self.dtype)
            self.normalize(p=2)
        else:
            self.tensor = tensor
        self.tensor.to(device=self.device, dtype=self.dtype)

    def act_single_gate(self, gate, pos, pos_control=None):
        if bf.compare_iterables(pos, pos_control):
            sys.exit('warning in act_single_gate: repeated '
                     'position(s) in pos and pos_control...')
        if pos_control is None:
            pos_control = []
        if gate.dtype is tc.complex128:
            self.tensor = self.tensor.to(dtype=tc.complex128)
        elif self.tensor.dtype is tc.complex128:
            gate = gate.to(dtype=tc.complex128)
        m_p = len(pos)
        m_c = len(pos_control)
        n_qubit = self.tensor.ndimension()
        shape = self.tensor.shape
        perm = list(range(n_qubit))
        for pp in pos:
            perm.remove(pp)
        for pp in pos_control:
            perm.remove(pp)
        perm = pos + perm + pos_control
        state1 = self.tensor.permute(perm).reshape(
            2 ** m_p, -1, 2 ** m_c)
        state1_ = state1[:, :, :-1]
        state2_ = gate.reshape(-1, 2 ** m_p).mm(
            state1[:, :, -1])
        state1 = tc.cat([state1_, state2_.reshape(
            state2_.shape + (1,))], dim=-1)
        state1 = state1.reshape(shape)
        self.tensor = state1.permute(
            bf.inverse_permutation(perm))

    def bipartite_ent(self, pos):
        ind = list(range(self.tensor.ndimension()))
        for x in pos:
            ind.remove(x)
        psi = self.tensor.permute(pos+ind).reshape(2**len(pos), -1)
        return tc.linalg.svdvals(psi)

    def bipartite_ent_entropy_all(self, eps=1e-14):
        ent = tc.zeros(self.tensor.ndimension()-1,
                       device=self.device, dtype=self.dtype)
        for n in range(1, self.tensor.ndimension()):
            tensor = self.tensor.reshape(np.prod(self.tensor.shape[:n]), -1)
            lm = tc.linalg.svdvals(tensor) ** 2
            ent[n-1] = -lm.inner(tc.log(lm+eps))
        return ent

    def onsite_ent_entropy(self, eps=1e-14):
        OEE = tc.zeros(self.tensor.ndimension(), device=self.device, dtype=self.dtype)
        for n in range(self.tensor.ndimension()):
            rho = self.reduced_density_matrix(n)
            lm = tc.linalg.eigvalsh(rho)
            OEE[n] = -lm.inner(tc.log(lm+eps))
        return OEE

    def normalize(self, p=2):
        norm = self.tensor.norm(p=p)
        self.tensor /= norm
        return norm

    def observation(self, operator, pos):
        rho = self.reduced_density_matrix(pos)
        d = self.tensor.shape[0]
        if operator.ndimension() > 2.5:
            dim_h = d ** int(operator.ndimension() / 2)
        else:
            dim_h = operator.shape[0]
        return tc.trace(rho.mm(operator.reshape(dim_h, dim_h)))

    def observe_bond_energies(self, hamilt, pos):
        eb = tc.zeros(len(pos), device=self.device, dtype=self.dtype)
        if type(hamilt) is np.ndarray:
            hamilt = tc.tensor(hamilt).to(device=self.device, dtype=self.dtype)
        if type(hamilt) is tc.Tensor:
            for p in range(len(pos)):
                eb[p] = self.observation(hamilt, pos[p])
        else:
            for p in range(len(pos)):
                eb[p] = self.observation(hamilt[p], pos[p])
        return eb

    def project(self, v, pos, update_state=True, return_tensor=False):
        psi1 = tc.tensordot(self.tensor, v, [[pos], [0]])
        p = psi1.norm()
        psi1 /= p
        if update_state:
            self.tensor = psi1
        if return_tensor:
            return psi1
        else:
            return TensorPureState(psi1.clone())

    def reduced_density_matrix(self, pos):
        if type(pos) is int:
            ind = list(range(self.tensor.ndimension()))
            ind.remove(pos)
            return tc.tensordot(self.tensor.conj(), self.tensor, (ind, ind)).T
        else:
            ind = list(range(self.tensor.ndimension()))
            dim = 1
            for n in pos:
                ind.remove(n)
                dim *= self.tensor.shape[n]
            x = self.tensor.permute(pos + ind).reshape(dim, -1)
        return x.mm(x.t().conj())

    def sampling(self, n_shots=1024, position=None, basis=None,
                 if_print=False, rand_seed=None):
        if rand_seed is None:
            rand_seed = self.rand_seed
        if self.rand_seed is not None:
            random.seed(rand_seed)
        if position is None:
            position = list(range(self.tensor.ndimension()))
        if basis is None:
            basis = ['z'] * len(position)

        mats = list()
        pos_xy = list()
        pauli_basis = mf.pauli_basis(device=self.device, if_list=False)
        for n in range(len(position)):
            if basis[n] in ['x', 'y']:
                mats.append(pauli_basis[basis[n]])
                pos_xy.append(position[n])
        state_ = mf.tucker_product(self.tensor, mats, pos_xy, dim=0)

        flag_all_m = (len(position) == self.tensor.ndimension())
        if flag_all_m:
            state_ = state_.reshape(-1, )
            weight = state_.dot(state_.conj())
        else:
            pos_ = list(range(self.tensor.ndimension()))
            dim = 1
            for x in position:
                pos_.remove(x)
                dim *= self.tensor.shape[x]
            state_ = state_.permute(pos_ + position).reshape(-1, dim)
            weight = tc.einsum('ab,ab->b', state_, state_.conj())

        population = bf.binary_strings(2 ** len(position))
        res = Counter(random.choices(population, weight, k=n_shots))
        if if_print:
            for key in res.keys():
                print(key, res[key])
        return res
