import torch as tc
import numpy as np
from Library import BasicFun as bf
from Library import MathFun as mf


def lazy_classifier(samples, samples_ref,
                    labels_ref, labels=None, para=None):
    para0 = {
        'kernel': 'euclidean',
        'beta': 1.02,
    }
    para = bf.combine_dicts(para0, para)
    para['kernel'] = para['kernel'].lower()

    classes = sorted(list(set(labels_ref.numpy())))
    prob = list()
    for c in classes:
        if para['kernel'] in ['cos-sin', 'cossin']:
            dis1 = mf.metric_neg_log_cos_sin(
                samples, samples_ref[labels_ref == c],
                average=True)
        elif para['kernel'] in ['r-cos-sin', 'rcossin']:
            dis1 = mf.metric_neg_log_cos_sin(
                samples, samples_ref[labels_ref == c],
                average=False)
            dis1 = (para['beta'] ** dis1
                    ).sum(dim=1) / dis1.shape[1]
        elif para['kernel'] == 'chebyshev':
            dis1 = mf.metric_neg_chebyshev(
                samples, samples_ref[labels_ref == c])
        elif para['kernel'] == 'cossin-chebyshev':
            dis1 = mf.metric_neg_cossin_chebyshev(
                samples, samples_ref[labels_ref == c])
        else:  # 欧式核
            dis1 = mf.metric_euclidean(
                samples, samples_ref[labels_ref == c])
        prob.append(dis1.reshape(dis1.shape[0], 1))
    prob = tc.cat(prob, dim=1)
    pred = prob.argmin(dim=1)
    if labels is not None:
        acc = (pred == labels).sum() / len(labels)
    else:
        acc = None
    return acc, pred


def knn_classifier(k, samples, samples_ref, labels_ref, labels=None):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(samples_ref.reshape(samples_ref.shape[0], -1), labels_ref)
    pred = knn.predict(samples.reshape(samples.shape[0], -1))
    if labels is not None:
        acc = np.sum(np.array(pred) == labels) / len(labels)
    else:
        acc = None
    return acc, pred
