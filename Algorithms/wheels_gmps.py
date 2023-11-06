import os
import torch as tc
import numpy as np
from random import choices
from Library.BasicFun import load, choose_device
from Library.DataFun import feature_map
from Library.MatrixProductState import generative_MPS


def acc_from_nll(nll, labels):
    pred = tc.argmin(nll, dim=1)
    num_c = tc.sum(tc.eq(pred, labels.to(device=nll.device)))
    return num_c.to(dtype=tc.float64) / labels.numel()


def acc_from_saved_gmps(samples, labels, para, paraMPS, batch_size=5000, pos=None):
    # pos: 用于实现特征选择后的nll计算
    classes = sorted(list(set(list(labels.cpu().numpy()))))
    gmps_names = list()
    for digit in range(len(classes)):
        gmps_names.append(os.path.join(para['save_dir'], gmps_save_name1(classes[digit], paraMPS)))
    return acc_saved_gmpsc_by_file_names(samples, labels, gmps_names, batch_size=batch_size, pos=pos)


def acc_saved_gmpsc_by_file_names(samples, labels, gmps_files, batch_size=5000,
                                  pos=None, classes=None):
    # pos: 用于实现特征选择后的nll计算
    if classes is None:
        if labels.ndimension() == 0:
            classes = [labels.item()]
        else:
            classes = sorted(list(set(list(labels.cpu().numpy()))))
    samples = samples.reshape(samples.shape[0], -1)

    nll, nll_tmp = list(), list()
    for digit in range(len(classes)):
        tensors, para, paraMPS = load(gmps_files[digit])
        mps = generative_MPS(tensors=tensors, para=paraMPS)
        mps.correct_device()
        samples_tmp = samples.clone().to(device=mps.device, dtype=mps.dtype)
        nll_tmp = list()
        while samples_tmp.numel() > 0:
            samples_ = samples_tmp[:batch_size]
            samples_ = feature_map(samples_, paraMPS['feature_map'],
                                   {'d': paraMPS['d'], 'theta': paraMPS['theta']})
            if pos is None:
                nll_now = mps.evaluate_nll(samples_)
            elif type(pos) in [list, tuple] and len(pos) == len(classes):
                nll_now = mps.evaluate_nll_selected_features(samples_, pos[digit])
            else:  # 所有GMPS用同一个pos
                nll_now = mps.evaluate_nll_selected_features(samples_, pos)
            nll_tmp.append(nll_now.reshape(nll_now.numel(), 1))
            tc.cuda.empty_cache()
            samples_tmp = samples_tmp[batch_size:]
        nll.append(tc.cat(nll_tmp, dim=0))
    return acc_from_nll(tc.cat(nll, dim=1), labels)


def acc_saved_gmpsc_FS_by_file_names(samples, labels, gmps_files, classes=None, batch_size=5000):
    # pos: 用于实现特征选择后的nll计算
    # GMPS仅用选择后的特征训练
    if classes is None:
        if labels.ndimension() == 0:
            classes = [labels.item()]
        else:
            classes = sorted(list(set(list(labels.cpu().numpy()))))
    if samples.ndimension() == 1:
        samples = samples.reshape(1, samples.numel())
    # samples = samples.reshape(samples.shape[0], -1)

    nll, nll_tmp = list(), list()
    for digit in range(len(classes)):
        data = load(gmps_files[digit], return_tuple=False)
        assert 'pos_fs' in data
        tensors, para, paraMPS, pos_fs = tuple(data[x] for x in data)
        samples_tmp = samples[:, pos_fs].clone()
        assert len(tensors) == len(pos_fs)
        mps = generative_MPS(tensors=tensors, para=paraMPS)
        mps.correct_device()
        nll_tmp = list()
        while samples_tmp.numel() > 0:
            samples_ = samples_tmp[:batch_size]
            samples_ = feature_map(samples_, paraMPS['feature_map'],
                                   {'d': paraMPS['d'], 'theta': paraMPS['theta']})
            nll_now = mps.evaluate_nll(samples_, update_vecs=False)
            nll_tmp.append(nll_now.reshape(nll_now.numel(), 1))
            tc.cuda.empty_cache()
            samples_tmp = samples_tmp[batch_size:]
        nll.append(tc.cat(nll_tmp, dim=0))
    return acc_from_nll(tc.cat(nll, dim=1), labels)


def gmps_save_name1(category, paraMPS):
    return 'GMPS_chi' + str(paraMPS['chi']) + '_theta' + str(paraMPS['theta']) \
           + '_FM_' + paraMPS['feature_map'] + '_digit_' + str(category)


def gmps_save_name2(category, paraMPS, dataset):
    return dataset + 'GMPS_chi' + str(paraMPS['chi']) + '_theta' + str(paraMPS['theta']) \
           + '_FM_' + paraMPS['feature_map'] + '_digit_' + str(category)


def generate_from_onebody_rho(rho, para_mps, para_g):
    sz = tc.tensor([[0, 0], [0, 1]],
                   dtype=para_mps['dtype'],
                   device=para_mps['device'])
    if para_g['way'] == 'mz':
        state = tc.trace(rho.mm(sz))
        vec = feature_map(
            state.view([1]), para_mps['feature_map'],
            para={'d': para_mps['d'],
                  'theta': para_mps['theta']})
        vec = vec.squeeze()
    elif para_g['way'] == 'inverse':
        state = tc.arccos(tc.sqrt(rho[0, 0])) / (
                para_mps['theta'] * np.pi / 2)
        vec = feature_map(
            state.view([1]), para_mps['feature_map'],
            para={'d': para_mps['d'],
                  'theta': para_mps['theta']})
        vec = vec.squeeze()
    else:
        lm = rho.diag()
        state = choices([0, 1], (
            lm).cpu().numpy(), k=1)[0]
        vec = state
    return state, vec
