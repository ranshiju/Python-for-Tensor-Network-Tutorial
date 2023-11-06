import torch as tc
from torch.utils.data import TensorDataset
from Library import BasicFun as bf
from Library import MathFun as mf
from Library import DataFun as df
from Algorithms.MPS_algo import GMPS_classification
from Algorithms.Kernel_methods import knn_classifier


def lazy_classifier(samples, samples_ref,
                    labels_ref, labels=None, para=None):
    # 该函数见/Algorithms/Kernel_methods.py
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
        acc = (pred == labels).sum() / labels.numel()
    else:
        acc = None
    return acc, pred


dataset = 'mnist'
classes = list(range(10))
num = 200  # 每类随机选出的样本数

train_dataset, test_dataset = df.load_mnist(
    dataset, process={'classes': classes})
samples, labels = df.dataset2tensors(train_dataset)
samples, labels = df.select_num_samples(
    samples, labels, num, classes)
samples_t, labels_t = df.dataset2tensors(test_dataset)
samples = samples.to(dtype=tc.float64)
samples_t = samples_t.to(dtype=tc.float64)

acc = knn_classifier(1, samples.numpy(), samples.numpy(),
                     labels.numpy(), labels.numpy())[0]
acc_t = knn_classifier(1, samples_t.numpy(), samples.numpy(),
                     labels.numpy(), labels_t.numpy())[0]
print('KNN (K=1) classifier:')
print('Train and Test accuracy = %g, %g' % (acc, acc_t))

acc = knn_classifier(200, samples.numpy(), samples.numpy(),
                     labels.numpy(), labels.numpy())[0]
acc_t = knn_classifier(200, samples_t.numpy(), samples.numpy(),
                     labels.numpy(), labels_t.numpy())[0]
print('KNN (K=200) classifier:')
print('Train and Test accuracy = %g, %g' % (acc, acc_t))

acc = lazy_classifier(
    samples, samples, labels, labels,
    {'kernel': 'chebyshev'})[0]
acc_t = lazy_classifier(
    samples_t, samples, labels, labels_t,
    {'kernel': 'chebyshev'})[0]
print('Lazy classifier with Chebyshev kernel:')
print('Train and Test accuracy = %g, %g' % (
    acc.item(), acc_t.item()))

acc = lazy_classifier(
    samples, samples, labels, labels,
    {'kernel': 'cossin-chebyshev'})[0]
acc_t = lazy_classifier(
    samples_t, samples, labels, labels_t,
    {'kernel': 'cossin-chebyshev'})[0]
print('Lazy classifier with Chebyshev NLF(cos-sin) kernel:')
print('Train and Test accuracy = %g, %g' % (
    acc.item(), acc_t.item()))

acc = lazy_classifier(
    samples, samples, labels, labels,
    {'kernel': 'euclidean'})[0]
acc_t = lazy_classifier(
    samples_t, samples, labels, labels_t,
    {'kernel': 'euclidean'})[0]
print('Lazy classifier with euclidean kernel:')
print('Train and Test accuracy = %g, %g' % (
    acc.item(), acc_t.item()))

acc = lazy_classifier(
    samples, samples, labels, labels,
    {'kernel': 'cossin'})[0]
acc_t = lazy_classifier(
    samples_t, samples, labels, labels_t,
    {'kernel': 'cossin'})[0]
print('Lazy classifier with NLF cos-sin kernel:')
print('Train and Test accuracy = %g, %g' % (
    acc.item(), acc_t.item()))

paraMPS={'d': 2, 'chi': 10}
acc_mps = GMPS_classification(
    TensorDataset(samples, labels),
    TensorDataset(samples_t, labels_t),
    para={'sweepTime': 30},
    paraMPS=paraMPS)
print('Train and test accuracy of GMPSC '
      '(d=%i, chi=%i) = %g and %g' % (
    paraMPS['d'], paraMPS['chi'], acc_mps[0], acc_mps[1]))
