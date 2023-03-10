import torch as tc
import numpy as np


def full_tensor(tensors):
    # 注：要求每个张量第0个指标为左虚拟指标，最后一个指标为右虚拟指标
    psi = tensors[0]
    for n in range(1, len(tensors)):
        psi = tc.tensordot(psi, tensors[n], [[-1], [0]])
    return psi


def mps_norm_square(tensors, normalize=False, form='log'):
    # norm = <psi|psi>
    v = tc.eye(tensors[0].shape[0], dtype=tensors[0].dtype, device=tensors[0].device)
    norm_list = list()
    for n in range(len(tensors)):
        v = tc.einsum('ap,adb,pdq->bq', v, tensors[n], tensors[n])
        norm_list.append(v.norm())
        v = v / norm_list[-1]
        if normalize:
            tensors[n] = tensors[n] / tc.sqrt(norm_list[-1])
    if form == 'log':  # 返回模方的log
        norm = 0.0
        for x in norm_list:
            norm = norm + tc.log(x)
    elif form == 'list':  # 返回列表
        return norm_list
    else:  # 直接返回模方
        norm = 1.0
        for x in norm_list:
            norm = norm * x
    return norm



