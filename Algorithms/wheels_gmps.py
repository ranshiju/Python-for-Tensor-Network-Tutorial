import copy
import os
import torch as tc
from Library.BasicFun import load
from Library.DataFun import feature_map
from Library.MatrixProductState import generative_MPS


def acc_from_nll(nll, labels):
    pred = tc.argmin(nll, dim=1)
    num_c = tc.sum(tc.eq(pred, labels.to(device=nll.device)))
    return num_c.to(dtype=tc.float64) / labels.numel()


def acc_from_saved_gmps(samples, labels, para, paraMPS, batch_size=1000):
    classes = sorted(list(set(list(labels.cpu().numpy()))))
    samples = samples.reshape(samples.shape[0], -1)

    nll, nll_tmp = list(), list()
    for digit in range(len(classes)):
        samples_tmp = samples.clone()
        para['save_name'] = gmps_save_name1(classes[digit], paraMPS)
        tensors = load(os.path.join(para['save_dir'], para['save_name']), 'tensors')
        if tensors is None:
            os.error('GMPS for category %i does not exist.')
        mps = generative_MPS(tensors=tensors, para=paraMPS)
        nll_tmp = list()
        while samples_tmp.numel() > 0:
            samples_ = samples_tmp[:batch_size]
            samples_ = feature_map(samples_, paraMPS['feature_map'],
                                   {'d': paraMPS['d'], 'theta': paraMPS['theta']})
            nll_now = mps.evaluate_nll(samples_)
            nll_tmp.append(nll_now.reshape(nll_now.numel(), 1))
            tc.cuda.empty_cache()
            samples_tmp = samples_tmp[batch_size:]
        nll.append(tc.cat(nll_tmp, dim=0))
    return acc_from_nll(tc.cat(nll, dim=1), labels)


def gmps_save_name1(category, paraMPS):
    return 'GMPS_chi' + str(paraMPS['chi']) + '_theta' + str(paraMPS['theta']) \
           + '_FM_' + paraMPS['feature_map'] + '_digit_' + str(category)
