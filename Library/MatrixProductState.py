import copy

import numpy as np
import torch as tc
from torch import nn
import Library.BasicFun as bf
import Library.DataFun as df

mps_propertie_keys = ['oee', 'eosp_ordering', 'eosp_oee_av', 'qs_number']


class MPS_basic:

    def __init__(self, tensors=None, para=None, properties=None):
        self.name = 'MPS'
        self.para = dict()
        self.input_paras(para)
        self.center = -1  # 正交中心（-1代表不具有正交中心）

        # 以下属性参考self.update_properties（通过properties输入）
        self.oee = None  # 单点纠缠熵（所有点）
        self.eosp_ordering = None  # EOSP采样顺序（所有点）
        self.eosp_oee_av = None  # EOSP采样顺序每次的平均OEE
        self.qs_number = None  # Q稀疏性

        self.eps = self.para['eps']
        self.device = bf.choose_device(self.para['device'])
        self.dtype = self.para['dtype']
        if tensors is None:
            self.tensors = random_mps(self.para['length'], self.para['d'], self.para['chi'],
                                      self.para['boundary'], self.device, self.dtype)
        else:
            self.tensors = tensors
            self.length = len(self.tensors)
            self.to()
        self.update_attributes_para()
        self.update_properties(properties)

    def input_paras(self, para=None):
        para0 = {
            'length': 4,
            'd': 2,
            'chi': 3,
            'boundary': 'open',
            'eps': 0,
            'device': None,
            'dtype': tc.float64
        }
        if para is None:
            self.para = para0
        else:
            self.para = dict(para0, **para)

    def add_mps_vecs(self, vecs, factors=1.0):
        # MPS is with open boundary condition
        # vecs.shape = (num_samples, d, num_qubits)
        assert self.para['boundary'] == 'open'
        tensors1 = from_vecs_to_mps(vecs, factors=factors)
        s = self.tensors[0].shape
        x1 = tc.zeros((1, s[1], s[2] + tensors1[0].shape[-1]),
                      dtype=self.tensors[0].dtype,
                      device=self.tensors[0].device)
        x1[0, :, :s[2]] = self.tensors[0]
        x1[0, :, s[2]:] = tensors1[0][0, :, :]
        self.tensors[0] = x1
        for n in range(1, len(self.tensors) - 1):
            s = self.tensors[n].shape
            x1 = tc.zeros((s[0] + tensors1[n].shape[0], s[1],
                           s[2] + tensors1[n].shape[2]),
                          dtype=self.tensors[n].dtype,
                          device=self.tensors[n].device)
            x1[:s[0], :, :s[2]] = self.tensors[n]
            x1[s[0]:, :, s[2]:] = tensors1[n]
            self.tensors[n] = x1
        s = self.tensors[-1].shape
        x1 = tc.zeros((s[0] + tensors1[-1].shape[0], s[1], 1),
                      dtype=self.tensors[-1].dtype,
                      device=self.tensors[-1].device)
        x1[:s[0], :, :1] = self.tensors[-1]
        x1[s[0]:, :, :1] = tensors1[-1]
        self.tensors[-1] = x1
        self.center = -1

    def copy_properties(self, mps, properties=None):
        if properties is None:
            properties = mps_propertie_keys
        for x in properties:
            setattr(self, x, getattr(mps, x))

    def correct_device(self):
        self.device = bf.choose_device(self.device)
        self.to()

    def clone_mps(self):
        tensors = [x.clone() for x in self.tensors]
        mps1 = MPS_basic(tensors=tensors, para=copy.deepcopy(self.para))
        for x in mps_propertie_keys:
            setattr(mps1, x, getattr(self, x))
        return mps1

    def clone_gmps(self):
        tensors = [x.clone() for x in self.tensors]
        mps1 = generative_MPS(tensors=tensors, para=copy.deepcopy(self.para))
        for x in mps_propertie_keys:
            setattr(mps1, x, getattr(self, x))
        return mps1

    def clone_tensors(self):
        self.tensors = [x.clone() for x in self.tensors]

    def bipartite_entanglement(self, nt, normalize=False):
        # 从第nt个张量右边断开，计算纠缠
        # 计算过程中，会对MPS进行规范变换
        if self.center <= nt:
            self.center_orthogonalization(nt, 'qr', dc=-1, normalize=normalize)
            lm = tc.linalg.svdvals(self.tensors[nt].reshape(
                -1, self.tensors[nt].shape[-1]))
        else:
            self.center_orthogonalization(nt + 1, 'qr', dc=-1, normalize=normalize)
            lm = tc.linalg.svdvals(self.tensors[nt+1].reshape(
                self.tensors[nt+1].shape[0], -1))
        return lm

    def center_orthogonalization(self, c, way='svd', dc=-1, normalize=False):
        if c == -1:
            c = len(self.tensors) - 1
        if self.center < -0.5:
            self.orthogonalize_n1_n2(0, c, way, dc, normalize)
            self.orthogonalize_n1_n2(len(self.tensors)-1, c, way, dc, normalize)
        elif self.center != c:
            self.orthogonalize_n1_n2(self.center, c, way, dc, normalize)
        self.center = c
        if normalize:
            self.normalize_central_tensor()

    def check_center_orthogonality(self, prt=True):
        if self.center < -0.5:
            if prt:
                print('MPS NOT in center-orthogonal form!')
        else:
            err = check_center_orthogonality(self.tensors, self.center, prt=prt)
            return err

    def check_virtual_dims(self):
        for n in range(len(self.tensors)-1):
            assert self.tensors[n].shape[-1] == self.tensors[n+1].shape[0]
        assert self.tensors[0].shape[0] == self.tensors[-1].shape[-1]

    def EOSP(self, num_f=-1, recalculate=False,
             clone=True, eps=1e-14):
        if clone:
            mps1 = self.clone_gmps()
            mps1.eps = eps
            eosp_ordering = mps1.EOSP_self(num_f, recalculate)
            self.copy_properties(mps1)
            return eosp_ordering
        else:
            self.eps = eps
            return self.EOSP_self(num_f, recalculate)

    def EOSP_average_OEEs(self, num_f=-1, recalculate=False,
                          clone=True, eps=1e-14):
        if (self.eosp_oee_av is not None) and (self.eosp_oee_av.numel() >= num_f):
            return self.eosp_oee_av[:num_f]
        else:
            if clone:
                mps1 = self.clone_gmps()
                mps1.eps = eps
                mps1.EOSP_self(num_f, recalculate)
                self.copy_properties(mps1)
                return mps1.eosp_oee_av
            else:
                self.eps = eps
                self.EOSP_self(num_f, recalculate)
                return self.eosp_oee_av

    def EOSP_fast_eps(self, num_f=-1, oee_eps=1e-6, eps=1e-14):
        # !!!! 未调试 !!!!
        self.eps = eps  # avoid NAN in OEE
        pos_eosp = list()
        if num_f == -1:
            num_f = len(self.tensors)
        pos_rec = list(range(len(self.tensors)))
        pos_large_oee = pos_rec
        qs_num = 0.0
        for n in range(num_f):
            bf.print_progress_bar(n, num_f)
            if len(pos_rec) == 1:
                pos_eosp.append(pos_rec[0])
            else:
                self.center_orthogonalization(
                    c=0, way='qr', normalize=True)
                OEE = self.entanglement_ent_onsite(which=pos_large_oee, print_info=False)
                if len(pos_large_oee) == 0:
                    order1 = tc.argsort(OEE, descending=True)
                    pos1 = [pos_rec[x] for x in order1[:(num_f-len(pos_eosp))]]
                    pos_eosp = pos_eosp + pos1
                    return pos_eosp, qs_num
                qs_num += OEE.sum().item() / OEE.numel() / np.log(
                    self.tensors[0].shape[1])
                p_max = OEE.argmax()
                pos_eosp.append(pos_rec.pop(p_max))
                rho = self.one_body_RDM(p_max)
                lm, u = tc.linalg.eigh(rho)
                u = u[:, lm.argmax()]
                self.project_multi_qubits(
                    [p_max], u.reshape(-1, 1))
                pos_large_oee = [x for x in range(OEE.numel()) if OEE[x] > oee_eps]
                pos_large_oee.pop(p_max)
        pos_eosp = tc.tensor(
            pos_eosp, dtype=tc.int64, device=self.device)
        return pos_eosp, qs_num

    def EOSP_self(self, num_f=-1, recalculate=False):
        pos_eosp = list()
        length = len(self.tensors)
        if num_f == -1:
            num_f = len(self.tensors)
        if recalculate or (self.eosp_ordering is None) \
                or self.eosp_ordering.numel() != len(self.tensors):
            pos_rec = list(range(len(self.tensors)))
            qs_num = 0.0
            eosp_oee_av = list()
            for n in range(num_f):
                bf.print_progress_bar(n, num_f)
                if len(pos_rec) <= 1:
                    pos_eosp.append(pos_rec[0])
                else:
                    self.center_orthogonalization(
                        c=0, way='qr', normalize=True)
                    OEE = self.entanglement_ent_onsite(
                        print_info=False)
                    eosp_oee_av.append(
                        OEE.sum().item() / OEE.numel())
                    qs_num += eosp_oee_av[-1] / np.log(
                        self.tensors[0].shape[1])
                    p_max = OEE.argmax()
                    pos_eosp.append(pos_rec.pop(p_max))
                    rho = self.one_body_RDM(p_max)
                    lm, u = tc.linalg.eigh(rho)
                    u = u[:, lm.argmax()]
                    self.project_multi_qubits(
                        [p_max], u.reshape(-1, 1))
            pos_eosp = tc.tensor(
                pos_eosp, dtype=tc.int64, device=self.device)
            eosp_oee_av = tc.tensor(
                eosp_oee_av, dtype=self.dtype, device=self.device)
            if pos_eosp.numel() == length:
                self.eosp_ordering = pos_eosp
                self.eosp_oee_av = eosp_oee_av
                self.qs_number = qs_num
        else:
            print('Use existing EOSP ordering...')
            pos_eosp = self.eosp_ordering[:num_f]
        return pos_eosp

    def entanglement_ent_onsite(self, which=None, eps=None,
                                recalculate=False, print_info=True):
        if which in [None, 'all'] and (not recalculate):
            if (self.oee is not None) and (self.oee.numel() == len(self.tensors)):
                if print_info:
                    print('Return existing OEE data...')
                return self.oee.to(device=self.device, dtype=self.dtype)
        if type(which) not in [tc.Tensor, np.ndarray, list, tuple] \
                and (which in [None, 'all']):
            which = list(range(len(self.tensors)))
        if eps is None:
            eps = self.eps
        if type(which) is int:
            which = [which]
        rho = list()
        for nc in which:
            self.center_orthogonalization(nc)
            rho.append(self.one_body_RDM(self.center).to('cpu').numpy())
        rho = np.stack(rho)
        lms = np.linalg.eigvalsh(rho)
        lms = tc.from_numpy(lms).to(device=self.device, dtype=self.dtype)
        lms = tc.einsum('na,n->na', lms, 1/lms.sum(dim=1))
        lms[lms < eps] = eps
        ent = tc.einsum('na,na->n', -lms, tc.log(lms))
        if ent.numel() == len(self.tensors):
            self.oee = ent
        return ent

    def find_max_virtual_dim(self):
        dims = [x.shape[0] for x in self.tensors]
        return max(dims)

    def full_tensor(self):
        return full_tensor(self.tensors)

    def inner(self, tensors, form='log'):
        if type(tensors) is list:
            return inner_product(self.tensors, tensors, form=form)
        else:
            return inner_product(self.tensors, tensors.tensors, form=form)

    def move_center_one_step(self, direction, decomp_way, dc, normalize):
        if direction.lower() in ['right', 'r']:
            if -0.5 < self.center < self.length-1:
                self.orthogonalize_left2right(self.center, decomp_way, dc, normalize)
                self.center += 1
            else:
                print('Error: cannot move center left as center = ' + str(self.center))
        elif direction.lower() in ['left', 'l']:
            if self.center > 0:
                self.orthogonalize_right2left(self.center, decomp_way, dc, normalize)
                self.center -= 1
            else:
                print('Error: cannot move center right as center = ' + str(self.center))

    def normalize_central_tensor(self, normalize=True):
        norm = self.tensors[self.center].norm()
        if normalize:
            self.tensors[self.center] = self.tensors[self.center] / norm
        return norm

    def norm_square(self, normalize=False, form='inner'):
        if self.center > -0.5:
            norm = self.normalize_central_tensor(normalize=normalize)
            if form == 'inner':
                norm = norm ** 2
            else:
                norm = 2 * tc.log(norm)
        else:
            norm, self.tensors = norm_square(self.tensors, normalize=normalize, form=form)
        return norm

    def normalize(self):
        norm, self.tensors = norm_square(self.tensors, normalize=True, form='list')
        return norm

    def one_body_RDM(self, nt):
        """
        :param nt: 计算第nt个自旋对应的单体约化密度矩阵
        :return rho: 约化密度矩阵
        """
        if self.center < -0.5:
            # 这种情况下，MPS不具备中心正交形式
            vl = tc.ones((1, 1), device=self.device, dtype=self.dtype)
            for n in range(nt):
                vl = tc.einsum('apb,cpd,ac->bd',
                               self.tensors[n].conj(), self.tensors[n], vl)
                vl = vl / vl.norm()
            vr = tc.ones((1, 1), device=self.device, dtype=self.dtype)
            for n in range(self.length-1, nt, -1):
                vr = tc.einsum('apb,cpd,bd->ac',
                               self.tensors[n].conj(), self.tensors[n], vl)
                vr = vr / vr.norm()
            rho = tc.einsum('apb,cqd,ac,bd->pq',
                            self.tensors[nt].conj(), self.tensors[nt], vl, vr)
        else:
            if self.center < nt:
                v = tc.eye(self.tensors[self.center].shape[0],
                           device=self.device, dtype=self.dtype)
                for n in range(self.center, nt):
                    v = tc.einsum('apb,cpd,ac->bd',
                                  self.tensors[n].conj(), self.tensors[n], v)
                    v = v / v.norm()
                rho = tc.einsum('apb,cqb,ac->pq',
                                self.tensors[nt].conj(),
                                self.tensors[nt], v)
            else:
                v = tc.eye(self.tensors[self.center].shape[-1],
                           device=self.device, dtype=self.dtype)
                for n in range(self.center, nt, -1):
                    v = tc.einsum('apb,cpd,bd->ac',
                                  self.tensors[n].conj(), self.tensors[n], v)
                    v = v / v.norm()
                rho = tc.einsum('apb,aqd,bd->pq',
                                self.tensors[nt].conj(),
                                self.tensors[nt], v)
        return rho / tc.trace(rho)

    def orthogonalize_left2right(self, nt, way, dc=-1,
                                 normalize=False):
        # dc=-1意味着不进行裁剪
        assert nt < len(self.tensors)-1
        s = self.tensors[nt].shape
        if 0 < dc < s[-1]:
            # In this case, truncation is required
            way = 'svd'
            if_trun = True
        else:
            if_trun = False

        tensor = self.tensors[nt].reshape(-1, s[-1]).to('cpu')
        if way.lower() == 'svd':
            u, lm, v = tc.linalg.svd(tensor,
                                     full_matrices=False)
            lm = lm.to(dtype=u.dtype)
            if if_trun:
                u = u[:, :dc].to(device=self.device)
                r = tc.diag(lm[:dc]).to(device=self.device).mm(
                    v[:dc, :].to(device=self.device))
            else:
                u = u.to(device=self.device)
                r = tc.diag(lm).to(device=self.device).mm(v.to(device=self.device))
        else:
            u, r = tc.linalg.qr(tensor)
            lm = None
            u, r = u.to(device=self.device), r.to(device=self.device)
        self.tensors[nt] = u.reshape(s[0], s[1], -1)
        if normalize:
            r /= tc.norm(r)
        self.tensors[nt+1] = tc.tensordot(
            r, self.tensors[nt+1], [[1], [0]])
        return lm

    def orthogonalize_right2left(self, nt, way, dc=-1, normalize=False):
        # dc=-1意味着不进行裁剪
        assert nt > 0
        s = self.tensors[nt].shape
        if 0 < dc < s[0]:
            # In this case, truncation is required
            way = 'svd'
            if_trun = True
        else:
            if_trun = False

        tensor = self.tensors[nt].reshape(s[0], -1).t().to('cpu')
        if way.lower() == 'svd':
            u, lm, v = tc.linalg.svd(tensor, full_matrices=False)
            lm = lm.to(dtype=u.dtype)
            if if_trun:
                u = u[:, :dc].to(device=self.device)
                r = tc.diag(lm[:dc]).to(device=self.device).mm(v[:dc, :].to(self.device))
            else:
                u = u.to(device=self.device)
                r = tc.diag(lm).to(device=self.device).mm(v.to(device=self.device))
        else:
            u, r = tc.linalg.qr(tensor)
            lm = None
            u, r = u.to(device=self.device), r.to(device=self.device)
        self.tensors[nt] = u.t().reshape(-1, s[1], s[2])
        if normalize:
            r /= tc.norm(r)
        self.tensors[nt - 1] = tc.tensordot(self.tensors[nt - 1], r, [[2], [1]])
        return lm

    def orthogonalize_n1_n2(self, n1, n2, way, dc, normalize):
        if n1 < n2:
            for nt in range(n1, n2, 1):
                self.orthogonalize_left2right(nt, way, dc, normalize)
        elif n1 > n2:
            for nt in range(n1, n2, -1):
                self.orthogonalize_right2left(nt, way, dc, normalize)

    def project_qubit_nt(self, nt, state):
        states_vecs = (type(state) is tc.Tensor) and (state.numel() == 2)
        if states_vecs:
            self.tensors[nt] = tc.tensordot(self.tensors[nt], state, [[1], [0]])
        else:
            self.tensors[nt] = self.tensors[nt][:, state, :]
        if len(self.tensors) > 1:
            if nt == 0:  # contract to 1st tensor
                self.tensors[1] = tc.tensordot(self.tensors[0], self.tensors[1], [[1], [0]])
            else:  # contract to the left tensor
                self.tensors[nt-1] = tc.tensordot(self.tensors[nt-1], self.tensors[nt], [[-1], [0]])
            self.tensors.pop(nt)

    def project_multi_qubits(self, pos, states):
        # Not central-orthogonalized; not normalized
        assert type(pos) is list
        states_vecs = (type(states) is tc.Tensor) and (states.ndimension() == 2)
        for n, p in enumerate(pos):
            if states_vecs:
                self.tensors[p] = tc.tensordot(self.tensors[p], states[:, n], [[1], [0]])
            else:
                self.tensors[p] = self.tensors[p][:, states[n], :]
        pos = sorted(copy.deepcopy(pos))
        for p in pos[len(pos):0:-1]:
            assert p > 0
            self.tensors[p - 1] = tc.tensordot(self.tensors[p - 1], self.tensors[p], [[-1], [0]])
            self.tensors.pop(p)
        if len(self.tensors) > 1:
            if pos[0] == 0:
                self.tensors[1] = tc.tensordot(self.tensors[0], self.tensors[1], [[-1], [0]])
            else:
                self.tensors[pos[0]-1] = tc.tensordot(
                    self.tensors[pos[0]-1], self.tensors[pos[0]], [[-1], [0]])
            self.tensors.pop(pos[0])
        self.center = -1

    def properties(self, prop=None, which_prop=None):
        if prop is None:
            prop = dict()
        if which_prop is None:
            which_prop = mps_propertie_keys
        elif type(which_prop) is str:
            which_prop = [which_prop]
        for x in which_prop:
            if getattr(self, x) is not None:
                prop[x] = getattr(self, x)
        return prop

    def Q_sparsity(self, recalculate=False, clone=True, eps=1e-14):
        if recalculate or (self.qs_number is None):
            if clone:
                mps1 = self.clone_gmps()
                mps1.eps = eps
                self.eosp_ordering = mps1.EOSP_self(
                    num_f=-1, recalculate=recalculate)
                self.qs_number = mps1.qs_number
                return self.qs_number
            else:
                self.eps = eps
                self.EOSP_self(num_f=-1, recalculate=recalculate)
                return self.qs_number
        else:
            return self.qs_number

    def save(self, path=None, file=None):
        bf.save(path, file, [self.tensors, self.para, self.properties()],
                ['tensors', 'paraMPS', 'properties'])

    def save_properties(self, path=None, file=None, which_prop=None):
        pathfile = bf.join_path(path, file)
        prop = bf.load(pathfile, 'properties')
        prop = self.properties(prop, which_prop)
        bf.save(pathfile, None, data=[prop], names=['properties'], append=True)

    def tensors2ParameterList(self, eps=1.0):
        tensors = nn.ParameterList()
        for x in self.tensors:
            tensors.append(nn.Parameter(x * eps, requires_grad=True))
        self.tensors = tensors

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = bf.choose_device(device)
        if dtype is not None:
            self.dtype = dtype
        tensors = [x.to(device=self.device, dtype=self.dtype) for x in self.tensors]
        self.tensors = tensors

    def update_attributes_para(self):
        self.length = len(self.tensors)
        self.para['device'] = self.device
        self.para['dtype'] = self.dtype
        self.para['length'] = self.length

    def update_properties(self, properties):
        if type(properties) is dict:
            for x in mps_propertie_keys:
                if x in properties:
                    setattr(self, x, properties[x])


class ResMPS_basic(nn.Module, MPS_basic):

    def __init__(self, tensors=None, para=None):
        super(ResMPS_basic, self).__init__()
        para['boundary'] = 'periodic'
        MPS_basic.__init__(self, tensors=tensors, para=para)
        self.name = 'SimpleResMPS'
        self.input_paras_ResMPS(para)

        if self.para['last_fc']:
            self.fc = nn.Linear(
                self.para['chi'], self.para['classes'],
                bias=self.para['bias_fc']).to(
                device=self.device, dtype=self.dtype)
        else:
            self.fc = None

        self.pos_c = self.para['pos_c']
        if self.pos_c == 'mid':
            self.pos_c = round(len(self.tensors) / 2)
        if (tensors is None) or (
                self.tensors[self.pos_c].ndimension != 4):
            s = self.tensors[self.pos_c].shape
            if self.fc is None:
                self.tensors[self.pos_c] = tc.randn(
                    s[0], s[1], self.para['classes'], s[2]).to(
                    device=self.device, dtype=self.dtype)
            else:
                self.tensors[self.pos_c] = tc.randn(
                    s[0], s[1], self.para['chi'], s[2]).to(
                    device=self.device, dtype=self.dtype)

        self.to(self.device, self.dtype)
        self.tensors2ParameterList(self.para['eps'])
        if self.para['bias']:
            self.bias = nn.Parameter(
                self.para['eps'] * tc.randn((
                    self.length, self.para['chi']),
                    device=self.device, dtype=self.dtype
                )*self.para['eps'], requires_grad=True)
        else:
            self.bias = None
        self.dropout = self.para['dropout']
        self.update_attributes_para()

    def forward(self, x, vL=None, vR=None):
        # x.shape = [样本数，特征维数，特征数]
        if vL is None:
            vL = tc.ones((x.shape[0], self.tensors[0].shape[0]),
                         device=self.device, dtype=self.dtype)
            vL = vL / self.tensors[0].shape[0]
        for n in range(self.pos_c):
            dv = tc.einsum(
                'abc,na,nb->nc',
                self.tensors[n], vL, x[:, :, n])
            if self.bias is not None:
                dv = dv + self.bias[n, :].repeat(x.shape[0], 1)
            if self.dropout is not None:
                dv = nn.Dropout(p=self.dropout)(dv)
            vL = vL + dv

        if vR is None:
            vR = tc.ones((x.shape[0], self.tensors[-1].shape[-1]),
                         device=self.device, dtype=self.dtype)
            vR = vR / self.tensors[-1].shape[-1]
        for n in range(x.shape[2]-1, self.pos_c, -1):
            dv = tc.einsum(
                'abc,nc,nb->na',
                self.tensors[n], vR, x[:, :, n])
            if self.bias is not None:
                dv = dv + self.bias[n, :].repeat(x.shape[0], 1)
            if self.dropout is not None:
                dv = nn.Dropout(p=self.dropout)(dv)
            vR = vR + dv

        dv = tc.einsum('abcd,na,nb,nd->nc', self.tensors[
            self.pos_c], vL, x[:, :, self.pos_c], vR)

        if self.fc is None:
            if self.bias is not None:
                v = dv + self.bias[self.pos_c, :].repeat(
                    x.shape[0], 1)
            else:
                v = dv
        else:
            if self.bias is not None:
                dv = dv + self.bias[self.pos_c, :].repeat(
                    x.shape[0], 1)
            if self.dropout is not None:
                dv = nn.Dropout(p=self.dropout)(dv)
            v = dv + (vL + vR) / 2
            v = self.fc(v)
        return v

    def input_paras_ResMPS(self, para=None):
        para0 = {
            'pos_c': 'mid',
            'eps': 1e-2,
            'classes': 2,  # 类别数
            'bias': False,
            'dropout': None,
            'last_fc': False  # 是否在最后加一层FC层
        }
        if para is None:
            self.para = dict(self.para, **para0)
        else:
            self.para = dict(self.para, **dict(para0, **para))

    def tensors2ParameterList(self, eps=1.0):
        tensors = nn.ParameterList()
        for n, x in enumerate(self.tensors):
            if (self.fc is None) and (n == self.pos_c):
                tensors.append(nn.Parameter(x, requires_grad=True))
            else:
                tensors.append(nn.Parameter(x*eps, requires_grad=True))
        self.tensors = tensors


class activated_ResMPS(ResMPS_basic):

    def __init__(self, tensors=None, para=None):
        super(activated_ResMPS, self).__init__(
            tensors=tensors, para=para)
        self.name = 'ActivatedResMPS'
        self.input_paras_activated_ResMPS(para)
        if self.para['activation'] is not None:
            self.activate = eval('nn.' + self.para['activation'] + '()')
        else:
            self.activate = None

    def forward(self, x, vL=None, vR=None):
        # x.shape = [样本数，特征维数，特征数]
        if vL is None:
            vL = tc.ones((x.shape[0], self.tensors[0].shape[0]),
                         device=self.device, dtype=self.dtype)
            vL = vL / x.shape[1]
        for n in range(self.pos_c):
            dv = tc.einsum('abc,na,nb->nc', self.tensors[n], vL, x[:, :, n])
            if self.bias is not None:
                dv = dv + self.bias[n, :].repeat(x.shape[0], 1)
            if self.activate is not None:
                dv = self.activate(dv)
            if self.dropout is not None:
                dv = nn.Dropout(p=self.dropout)(dv)
            vL = vL + dv

        if vR is None:
            vR = tc.ones((x.shape[0], self.tensors[0].shape[0]),
                         device=self.device, dtype=self.dtype)
            vR = vR / x.shape[1]
        for n in range(x.shape[2]-1, self.pos_c, -1):
            dv = tc.einsum('abc,nc,nb->na', self.tensors[n], vR, x[:, :, n])
            if self.bias is not None:
                dv = dv + self.bias[n, :].repeat(x.shape[0], 1)
            if self.activate is not None:
                dv = self.activate(dv)
            if self.dropout is not None:
                dv = nn.Dropout(p=self.dropout)(dv)
            vR = vR + dv

        dv = tc.einsum('abcd,na,nb,nd->nc', self.tensors[
            self.pos_c], vL, x[:, :, self.pos_c], vR)
        if self.fc is None:
            if self.bias is not None:
                v = dv + self.bias[self.pos_c, :].repeat(x.shape[0], 1)
            else:
                v = dv
        else:
            if self.bias is not None:
                dv = dv + self.bias[self.pos_c, :].repeat(x.shape[0], 1)
            if self.activate is not None:
                dv = self.activate(dv)
            if self.dropout is not None:
                dv = nn.Dropout(p=self.dropout)(dv)
            v = dv + (vL + vR) / 2
            v = self.fc(v)
        return v

    def input_paras_activated_ResMPS(self, para=None):
        para0 = {
            'activation': 'ReLU'}
        if para is None:
            self.para = dict(self.para, **para0)
        else:
            self.para = dict(self.para, **dict(para0, **para))


class generative_MPS(MPS_basic):

    def __init__(self, tensors=None, para=None, properties=None):
        super(generative_MPS, self).__init__(
            tensors=tensors, para=para, properties=properties)
        self.samples_v = None
        self.vecsL = None
        self.vecsR = None
        self.norms = tc.ones(1)
        self.para = copy.deepcopy(para)
        self.combine_default_para()
        self.eps = self.para['eps']

    def clear_memory(self):
        self.samples_v = None
        self.vecsL = None
        self.vecsR = None
        self.norms = tc.ones(1)

    def combine_default_para(self):
        para = {
            'length': 784,
            'd': 2,
            'chi': 3,
            'boundary': 'open',
            'feature_map': 'cossin',
            'eps': 0,
            'theta': 1.0,
            'device': bf.choose_device(),
            'dtype': tc.float64
        }
        self.para = dict(para, **self.para)

    def evaluate_nll(self, samples=None, average=False, update_vecs=True):
        if samples is not None:
            if samples.ndimension() == 2:
                samples_v = df.feature_map(
                    samples, self.para['feature_map'],
                    {'d': self.para['d'], 'theta': self.para['theta']})
            else:
                assert samples.ndimension() == 3
                samples_v = samples
            self.input_samples_v(samples_v)
        if update_vecs:
            center = max(0, self.center)
            self.initialize_vecs_norms()
            for n in range(center):
                self.update_vecsL_n(n)
            for n in range(len(self.tensors) - 1, center, -1):
                self.update_vecsR_n(n)
            self.update_norms_center(center)
        else:  # 避免inplace操作
            vl = tc.ones((samples.shape[0], 1), device=self.device, dtype=self.dtype)
            vr = tc.ones((samples.shape[0], 1), device=self.device, dtype=self.dtype)
            self.norms = tc.ones(
                (samples.shape[0], len(self.tensors)),
                device=self.device, dtype=self.dtype)
            center = max(0, self.center)
            for n in range(center):
                vl = tc.einsum('na,nb,abc->nc', vl,
                               self.samples_v[:, :, n], self.tensors[n])
                self.norms[:, n] = vl.norm(dim=1)
                vl = tc.einsum(
                    'na,n->na', vl,
                    1 / (self.norms[:, n] + self.eps))
            for n in range(len(self.tensors) - 1, center, -1):
                vr = tc.einsum(
                    'nc,nb,abc->na', vr,
                    self.samples_v[:, :, n], self.tensors[n])
                self.norms[:, n] = vr.norm(dim=1)
                vr = tc.einsum(
                    'na,n->na', vr,
                    1 / (self.norms[:, n] + self.eps))
            self.norms[:, center] = tc.einsum(
                'apb,na,np,nb->n', self.tensors[center], vl,
                self.samples_v[:, :, center], vr)
        nll = self.evaluate_nll_from_norms(average=average)
        return nll

    def evaluate_nll_selected_features(self, samples_v, pos, average=False):
        # 注：samples_v包含所有特征，不是选出来的特征
        assert samples_v.shape[2] == len(self.tensors)
        assert type(pos) in [list, tuple, tc.Tensor]
        vl = tc.ones((samples_v.shape[0], 1, 1), device=self.device, dtype=self.dtype)
        vr = tc.ones((samples_v.shape[0], 1, 1), device=self.device, dtype=self.dtype)
        self.norms = tc.ones((samples_v.shape[0], len(self.tensors)),
                             device=self.device, dtype=self.dtype)
        center = max(0, self.center)
        for n in range(center):
            if n in pos:
                mat = tc.einsum('asb,ns->nab', self.tensors[n], samples_v[:, :, n])
                vl = tc.einsum('nab,nac,ncd->nbd', mat.conj(), vl, mat)
            else:
                vl = tc.einsum('asb,csd,nac->nbd', self.tensors[n].conj(), self.tensors[n], vl)
            self.norms[:, n] = vl.reshape(vl.shape[0], -1).norm(dim=1)
            vl = tc.einsum(
                'nab,n->nab', vl,
                1 / (self.norms[:, n] + self.eps))
        for n in range(self.para['length'] - 1, center, -1):
            if n in pos:
                mat = tc.einsum('asb,ns->nab', self.tensors[n], samples_v[:, :, n])
                vr = tc.einsum('nab,nbd,ncd->nac', mat.conj(), vr, mat)
            else:
                vr = tc.einsum('asb,csd,nbd->nac', self.tensors[n].conj(), self.tensors[n], vr)
            self.norms[:, n] = vr.reshape(vr.shape[0], -1).norm(dim=1)
            vr = tc.einsum(
                'nab,n->nab', vr,
                1 / (self.norms[:, n] + self.eps))
        if center in pos:
            mat = tc.einsum('asb,ns->nab', self.tensors[center], samples_v[:, :, center])
            self.norms[:, center] = tc.einsum(
                'nab,nac,ncd,nbd->n', mat.conj(), vl, mat, vr).abs()
        else:
            self.norms[:, center] = tc.einsum(
                'asb,csd,nac,nbd->n', self.tensors[center].conj(),
                self.tensors[center], vl, vr).abs()
        nll = self.evaluate_nll_from_norms(average=average)
        return nll

    def evaluate_nll_from_norms(self, average=True):
        if average:
            return - 2 * tc.log(self.norms.abs() + self.eps
                                ).sum() / self.norms.shape[0]
        else:
            return - 2 * tc.log(self.norms.abs() + self.eps
                                ).sum(dim=1)

    def feature_selection_OEE(self, num, ent=None):
        if ent is None:
            ent = self.entanglement_ent_onsite(which='all')
        return tc.sort(ent, descending=True)[1][:num]

    def grad_update_MPS_tensor_by_env(self, lr, way='tsgo'):
        env = self.obtain_env_center()
        grad = 2 * self.tensors[self.center] - 2 * env
        if way.lower() == 'tsgo':
            grad = grad.reshape(-1, )
            proj = tc.dot(grad, self.tensors[self.center].reshape(
                -1, )) * self.tensors[self.center].reshape(-1, )
            grad = grad - proj
        grad /= grad.norm()
        self.tensors[self.center] -= (lr * grad.reshape(env.shape))

    def initialize_vecs_norms(self):
        num_f = self.samples_v.shape[0]
        chi = self.find_max_virtual_dim()
        self.norms = tc.ones(
            (num_f, len(self.tensors)),
            device=self.device, dtype=self.dtype)
        self.vecsL = tc.ones(
            (num_f, chi, len(self.tensors)),
            device=self.device, dtype=self.dtype)
        self.vecsR = tc.ones(
            (num_f, chi, len(self.tensors)),
            device=self.device, dtype=self.dtype)
        center = max(0, self.center)
        for n in range(center):
            self.update_vecsL_n(n)
        for n in range(len(self.tensors) - 1, center, -1):
            self.update_vecsR_n(n)
        self.update_norms_center(center)

    def input_samples_v(self, samples_v):
        self.samples_v = samples_v.to(device=self.device, dtype=self.dtype)

    def obtain_env_center(self, n=None):
        if n is None:
            n = self.center
        s = self.tensors[n].shape
        env = tc.einsum(
            'na,np,nb->napb',
            self.vecsL[:, :s[0], n],
            self.samples_v[:, :, n],
            self.vecsR[:, :s[2], n])
        c_env = tc.einsum(
            'apb,napb->n',
            self.tensors[n], env)
        # 保证被除的数不接近于0
        c_env += tc.sign(c_env) * self.eps
        env = tc.einsum(
            'napb,n->apb',
            env, 1 / c_env) / c_env.numel()
        return env

    def update_norms_center(self, center=None):
        if center is None:
            center = self.center
        s = self.tensors[center].shape
        self.norms[:, center] = tc.einsum(
            'apb,na,np,nb->n', self.tensors[center], self.vecsL[:, :s[0], center],
            self.samples_v[:, :, center], self.vecsR[:, :s[2], center])

    def update_para(self, para):
        self.para = dict(self.para, **para)

    def update_vecsL_n(self, n):
        s = self.tensors[n].shape
        self.vecsL[:, :s[2], n + 1] = tc.einsum(
            'na,nb,abc->nc', self.vecsL[:, :s[0], n],
            self.samples_v[:, :, n], self.tensors[n])
        self.norms[:, n] = self.vecsL[:, :s[2], n + 1
                           ].norm(dim=1)
        self.vecsL[:, :s[2], n + 1] = tc.einsum(
            'na,n->na', self.vecsL[:, :s[2], n + 1],
            1 / (self.norms[:, n] + self.eps))

    def update_vecsR_n(self, n):
        s = self.tensors[n].shape
        self.vecsR[:, :s[0], n - 1] = tc.einsum(
            'nc,nb,abc->na', self.vecsR[:, :s[2], n],
            self.samples_v[:, :, n], self.tensors[n])
        self.norms[:, n] = self.vecsR[:, :s[0], n - 1
                           ].norm(dim=1)
        self.vecsR[:, :s[0], n - 1] = tc.einsum(
            'na,n->na', self.vecsR[:, :s[0], n - 1],
            1 / (self.norms[:, n] + self.eps))

    @staticmethod
    def average_nll_from_norms(norms, eps):
        return - 2 * tc.log(norms.abs() + eps).sum() / norms.shape[0]

    @staticmethod
    def acc_from_nll(nll, labels):
        pred = tc.argmin(nll, dim=1)
        num_c = tc.sum(pred == labels.to(device=nll.device))
        return num_c.to(dtype=tc.float64) / labels.numel()


def check_center_orthogonality(tensors, center, prt=False):
    err = [None] * len(tensors)
    for n in range(center):
        s = tensors[n].shape
        tmp = tensors[n].reshape(-1, s[-1])
        tmp = tmp.t().conj().mm(tmp)
        err[n] = (tmp - tc.eye(tmp.shape[0], device=tensors[n].device,
                               dtype=tensors[n].dtype)).norm(p=1).item()
    for n in range(len(tensors)-1, center, -1):
        s = tensors[n].shape
        tmp = tensors[n].reshape(s[0], -1)
        tmp = tmp.mm(tmp.t().conj())
        err[n] = (tmp - tc.eye(tmp.shape[0], device=tensors[n].device,
                               dtype=tensors[n].dtype)).norm(p=1).item()

    if prt:
        print('Orthogonality check:')
        print('=' * 35)
        err_av = 0.0
        for n in range(len(tensors)):
            if err[n] is None:
                print('Site ' + str(n) + ':  center')
            else:
                print('Site ' + str(n) + ': ', err[n])
                err_av += err[n]
        print('-' * 35)
        print('Average error = %g' % (err_av / (len(tensors) - 1)))
        print('=' * 35)
    return err


def copy_mps_properties(mps_to, mps_from, properties=None):
    if properties is None:
        properties = mps_propertie_keys
    for x in properties:
        setattr(mps_to, x, getattr(mps_from, x))
    return mps_to


def evaluate_nll(mps, samples=None, average=False):
    if type(mps) in [tuple, list]:
        mps = generative_MPS(mps[0], mps[1])
    if samples is not None:
        if samples.ndimension() == 2:
            samples_v = df.feature_map(
                samples, mps.para['feature_map'],
                {'d': mps.para['d'], 'theta': mps.para['theta']})
        else:
            assert samples.ndimension() == 3
            samples_v = samples
        mps.input_samples_v(samples_v)
    mps.initialize_vecs_norms()
    nll = mps.evaluate_nll_from_norms(average=average)
    return nll


def full_tensor(tensors):
    # 注：要求每个张量第0个指标为左虚拟指标，最后一个指标为右虚拟指标
    psi = tensors[0]
    for n in range(1, len(tensors)):
        psi = tc.tensordot(psi, tensors[n], [[-1], [0]])
    if psi.shape[0] > 1:  # 周期边界
        psi = psi.permute([0, psi.ndimension()-1] + list(range(1, psi.ndimension()-1)))
        s = psi.shape
        psi = tc.einsum('aab->b', psi.reshape(s[0], s[1], -1))
        psi = psi.reshape(s[2:])
    else:
        psi = psi.squeeze()
    return psi


def from_vecs_to_mps(vecs, factors=1.0):
    if vecs.ndimension() == 2:  # vecs.shape = (d,num_qubits)
        vecs[:, 0] = vecs[:, 0] * factors
        return [vecs[:, n].reshape(1, vecs.shape[0], 1) for n in range(vecs.shape[1])]
    else:  # vecs.shape = (num_samples, d, num_qubits)
        assert vecs.ndimension() == 3
        num_s, d, num_q = vecs.shape
        if type(factors) not in [tc.Tensor, tc.nn.parameter.Parameter]:
            factors = tc.tensor(factors, device=vecs.device, dtype=vecs.dtype)
        if factors.numel() == 1:
            tensors = [(vecs[:, :, 0].t() * factors).reshape(1, d, num_s)]
        else:
            assert factors.numel() == num_s
            tensors = [tc.einsum('nd,n->dn', vecs[:, :, 0], factors).reshape(1, d, num_s)]
        for n in range(1, num_q-1):
            x = tc.zeros((num_s, d, num_s), device=vecs.device, dtype=vecs.dtype)
            for s in range(d):
                x[:, s, :] = tc.diag(vecs[:, s, n])
            tensors.append(x)
        tensors.append(vecs[:, :, -1].reshape(num_s, d, 1))
        return tensors


def inner_product(tensors0, tensors1, form='log'):
    assert tensors0[0].shape[0] == tensors0[-1].shape[-1]
    assert tensors1[0].shape[0] == tensors1[-1].shape[-1]
    assert len(tensors0) == len(tensors1)

    v0 = tc.eye(tensors0[0].shape[0], dtype=tensors0[0].dtype, device=tensors0[0].device)
    v1 = tc.eye(tensors1[0].shape[0], dtype=tensors0[0].dtype, device=tensors0[0].device)
    v = tc.kron(v0, v1).reshape([tensors0[0].shape[0], tensors1[0].shape[0],
                                 tensors0[0].shape[0], tensors1[0].shape[0]])
    norm_list = list()
    for n in range(len(tensors0)):
        v = tc.einsum('uvap,adb,pdq->uvbq', v, tensors0[n].conj(), tensors1[n])
        norm_list.append(v.norm())
        v = v / norm_list[-1]
    if v.numel() > 1:
        norm1 = tc.einsum('acac->', v)
        norm_list.append(norm1)
    else:
        norm_list.append(v[0, 0, 0, 0])
    if form == 'log':  # 返回模方的log，舍弃符号
        norm = 0.0
        for x in norm_list:
            norm = norm + tc.log(x.abs())
    elif form == 'list':  # 返回列表
        return norm_list
    else:  # 直接返回模方
        norm = 1.0
        for x in norm_list:
            norm = norm * x
    return norm


def norm_square(tensors, normalize=False, form='log'):
    # norm = <psi|psi>
    assert tensors[0].shape[0] == tensors[-1].shape[-1]
    v = tc.eye(tensors[0].shape[0]**2, dtype=tensors[0].dtype, device=tensors[0].device
               ).reshape([tensors[0].shape[0]] * 4)
    # v = tc.kron(v0, v0).reshape([tensors[0].shape[0]] * 4)
    norm_list = list()
    for n in range(len(tensors)):
        v = tc.einsum('uvap,adb,pdq->uvbq', v, tensors[n].conj(), tensors[n])
        norm_list.append(v.norm())
        v = v / norm_list[-1]
        if normalize:
            tensors[n] = tensors[n] / tc.sqrt(norm_list[-1])
    if v.numel() > 1:
        norm1 = tc.einsum('acac->', v)
        norm_list.append(norm1)
        if normalize:
            tensors[-1] = tensors[-1] / tc.sqrt(norm_list[-1])
    if form == 'log':  # 返回模方的log
        norm = 0.0
        for x in norm_list:
            norm = norm + tc.log(x)
    elif form == 'list':  # 返回列表
        return norm_list, tensors
    else:  # 直接返回模方
        norm = 1.0
        for x in norm_list:
            norm = norm * x
    return norm, tensors


def random_mps(length, d, chi, boundary='open', device=None, dtype=tc.float64):
    device = bf.choose_device(device)
    if boundary == 'open':
        tensors = [tc.randn((chi, d, chi), device=device, dtype=dtype)
                   for _ in range(length - 2)]
        return [tc.randn((1, d, chi), device=device, dtype=dtype)] + tensors + [
            tc.randn((chi, d, 1), device=device, dtype=dtype)]
    else:  # 周期边界MPS
        return [tc.randn((chi, d, chi), device=device, dtype=dtype)
                for _ in range(length)]






