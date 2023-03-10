import os
import cv2
import numpy as np
import torch as tc
from torch.utils.data import TensorDataset, DataLoader
from Library.BasicFun import choose_device, combine_dicts


def adjust_lr(optimizer, factor, cond=True):
    if cond:
        for group in optimizer.param_groups:
            group['lr'] = group['lr'] * factor
    return optimizer


def binarize_samples(samples):
    return tc.round(samples)


def choose_classes(data, labels, classes):
    shape = list(data.shape)
    data_ = list()
    labels_ = list()
    data = data.reshape(shape[0], -1)
    if type(classes) is int:
        classes = [classes]
    for n in range(len(classes)):
        data_.append(data[labels == classes[n]])
        labels_.append(tc.ones((
            data_[-1].shape[0],), device=labels.device, dtype=labels.dtype) * n)
    data_ = tc.cat(data_, dim=0)
    labels_ = tc.cat(labels_, dim=0)
    shape[0] = labels_.numel()
    return data_.reshape(shape), labels_


def choose_classes_dataset(dataset, classes, re_tensor=False):
    train_dataset = DataLoader(dataset, batch_size=dataset.data.shape[0], shuffle=False)
    train_samples, train_lbs = next(iter(train_dataset))
    train_samples, train_lbs = choose_classes(train_samples, train_lbs, classes)
    if re_tensor:
        return train_samples, train_lbs
    else:
        return TensorDataset(train_samples, train_lbs)


def feature_map(samples, which='cossin', para=None, norm_p=2):
    if which is None:
        return samples
    which = which.lower()
    if para is None:
        para = dict()
    para_ = {
        'd': 2,
        'theta': 1,
        'alpha': 2
    }
    para = combine_dicts(para_, para)
    samples = samples.reshape(-1, 1, samples[0].numel())
    if which in ['cossin', 'cos-sin', 'cos_sin']:
        img1 = tc.cat([tc.cos(samples * para['theta'] * np.pi / 2),
                       tc.sin(samples * para['theta'] * np.pi / 2)], dim=1)
        if norm_p == 1:
            img1 = img1 ** 2
        return img1
    elif which == 'linear':
        img1 = tc.cat([samples, 1 - samples], dim=1)
        if norm_p == 2:
            img1 = tc.sqrt(img1)
        return img1
    elif which == 'gaussian':
        return feature_map_gaussian_discretization(
            samples, norm_p=norm_p, d=para['d'], alpha=para['alpha'])
    elif which in ['square-linear', 'square_linear', 'squarelinear',
                   'normalized-linear', 'normalized_linear', 'normalizedlinear']:
        img1 = tc.cat([tc.sqrt(tc.abs(samples)), tc.sqrt(tc.abs(1 - samples))], dim=1)
        if norm_p == 1:
            img1 = img1 ** 2
        return img1
    elif which in ['one-hot', 'one_hot', 'onehot']:
        return feature_map_one_hot(samples, d=para['d'])
    else:
        print('Error: ' + which + ' is not a valid feature map')


def feature_map_gaussian_discretization(samples, d, alpha=5, norm_p=1):
    # Feature取值在0到1间（含）
    x = tc.linspace(1/d/2, 1-1/d/2, d).to(device=samples.device, dtype=samples.dtype)
    print(x)
    samples_ = samples.reshape(samples.shape[0], 1, -1)
    s_list = list()
    for n in range(d):
        s_list.append(tc.exp(-alpha * (samples_ - x[n]) ** 2))
    s_list = tc.cat(s_list, dim=1)
    norms = s_list.norm(dim=1, p=norm_p)
    s_list = tc.einsum('nab,nb->nab', s_list, 1/norms)
    return s_list


def feature_map_one_hot(samples, d, eps=1e-10):
    # Feature取值在0到1间（含）
    x = tc.linspace(1/d, 1, d).to(device=samples.device, dtype=samples.dtype)
    x[-1] += eps
    samples_ = samples.reshape(samples.shape[0], 1, -1)
    samples1 = tc.zeros(samples_.shape, device=samples.device, dtype=samples.dtype)
    samples1[samples_ <= x[0]] = 1.0
    s_list = [samples1]
    for n in range(1, d):
        samples1 = tc.zeros(samples_.shape, device=samples.device, dtype=samples.dtype)
        samples1[samples_ <= x[n]] = 1.0
        samples1[samples_ <= x[n-1]] = 0.0
        s_list.append(samples1)
    s_list = tc.cat(s_list, dim=1)
    return s_list


def get_batch_from_loader(loader, which, only_sample=False):
    for n, x in enumerate(loader):
        if n == which:
            if only_sample:
                return x[0]
            else:
                return x


def labels_rearrange(labels):
    labels1 = labels.clone()
    numbers = set(list(labels.reshape(-1, ).numpy()))
    numbers = sorted(list(numbers))
    for x in numbers:
        labels1[labels == x] = numbers.index(x)
    return labels1


def load_cifar10(which=10, dataset_path=None, test=True, process=None):
    from torchvision import datasets, transforms
    preprocess = [transforms.ToTensor()]
    if process is None:
        process = dict()
    if 'crop' in process:
        preprocess.append(transforms.CenterCrop(size=process['crop']))
    if 'resize' in process:
        preprocess.append(transforms.Resize(size=process['resize']))
    data_tf = transforms.Compose(preprocess)
    if dataset_path is None:
        paths = ['./Datasets', '../Datasets', '../../Datasets', '../../../Datasets', '../../../../Datasets']
        for x in paths:
            if os.path.isdir(x):
                dataset_path = x
    if dataset_path is None:
        dataset_path = './Datasets'
    test_dataset = None
    if which == 10:
        train_dataset = datasets.CIFAR10(
            root=dataset_path, train=True, transform=data_tf, download=True)
        if test:
            test_dataset = datasets.MNIST(root=dataset_path, train=False, transform=data_tf)
    else:
        train_dataset = datasets.CIFAR100(
            root=dataset_path, train=True, transform=data_tf, download=True)
        if test:
            test_dataset = datasets.FashionMNIST(root=dataset_path, train=False, transform=data_tf)
    return train_dataset, test_dataset


def load_iris(return_dict=False, return_tensor=True, device=None, dtype=tc.float64):
    from sklearn import datasets
    iris = datasets.load_iris()
    if return_tensor:
        device = choose_device(device)
        iris['data'] = tc.from_numpy(iris['data']).to(device=device, dtype=dtype)
        iris['target'] = tc.from_numpy(iris['target']).to(device=device, dtype=tc.int64)
    if return_dict:
        return iris
    else:
        samples = iris['data']
        targets = iris['target']
        return samples, targets


def load_mnist(which='mnist', dataset_path=None, test=True, process=None):
    from torchvision import datasets, transforms
    preprocess = [transforms.ToTensor()]
    if process is None:
        process = dict()
    if 'crop' in process:
        preprocess.append(transforms.CenterCrop(size=process['crop']))
    if 'resize' in process:
        preprocess.append(transforms.Resize(size=process['resize']))

    data_tf = transforms.Compose(preprocess)
    if dataset_path is None:
        paths = ['./Datasets', '../Datasets', '../../Datasets', '../../../Datasets', '../../../../Datasets']
        for x in paths:
            if os.path.isdir(x):
                dataset_path = x
                break
    if dataset_path is None:
        dataset_path = './Datasets'
    test_dataset = None
    if which.lower() == 'mnist':
        train_dataset = datasets.MNIST(
            root=dataset_path, train=True, transform=data_tf, download=True)
        if test:
            test_dataset = datasets.MNIST(root=dataset_path, train=False, transform=data_tf)
    else:
        train_dataset = datasets.FashionMNIST(
            root=dataset_path, train=True, transform=data_tf, download=True)
        if test:
            test_dataset = datasets.FashionMNIST(root=dataset_path, train=False, transform=data_tf)

    if 'classes' in process:
        train_dataset = choose_classes_dataset(train_dataset, process['classes'])
        if test:
            test_dataset = choose_classes_dataset(test_dataset, process['classes'])
    return train_dataset, test_dataset


def make_dataloader(dataset, batch_size=None, shuffle=False):
    from torch.utils.data import DataLoader
    if batch_size is None:
        batch_size = dataset.data.shape[0]
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def one_hot_labels(labels, device=None, dtype=None):
    if device is None:
        device = labels.device
    if dtype is None:
        dtype = tc.float64
    labels_value = sorted(list(set(list(labels.reshape(-1, ).numpy()))))
    num_c = len(labels_value)
    labels_v = tc.zeros((labels.shape[0], num_c), device=device, dtype=dtype)
    for n in range(labels.shape[0]):
        labels_v[n, labels_value.index(labels[n])] = 1.0
    return labels_v


def preprocess_image(image, preprocess_means):
    if 'cut' in preprocess_means:
        lx = min(preprocess_means['cut'][0], image.shape[0])
        ly = min(preprocess_means['cut'][1], image.shape[1])
        lx0 = int((image.shape[0] - lx) / 2)
        ly0 = int((image.shape[1] - ly) / 2)
        image = image[lx0:lx0 + lx, ly0:ly0 + ly, :]
    if 'size' in preprocess_means:
        if image.shape[0] > preprocess_means['size'][0] or image.shape[1] > preprocess_means['size'][1]:
            image = cv2.resize(image, preprocess_means['size'])
    return image


def rescale_max_min_simple(samples, maximum=1, minimum=0):
    s = samples.shape
    samples_ = samples - samples.min()
    samples_ = samples_ / samples_.max()
    samples_ = samples_ * (maximum - minimum) + minimum
    return samples_.reshape(s)


def rescale_max_min_sample_wise(samples, maximum=1, minimum=0):
    s = samples.shape
    samples_ = samples.reshape(samples.shape[0], -1)
    samples_ = samples_ - samples_.min(dim=1)[0].repeat(samples_.shape[1], 1).permute(1, 0)
    samples_max = samples_.max(dim=1)[0].repeat(samples_.shape[1], 1).permute(1, 0)
    samples_ = samples_ / samples_max
    samples_ = samples_ * (maximum - minimum) + minimum
    return samples_.reshape(s)


def rescale_max_min_feature_wise(samples, maximum=1, minimum=0):
    s = samples.shape
    samples_ = samples.reshape(samples.shape[0], -1)
    samples_ = samples_ - samples_.min(dim=0)[0].repeat(samples_.shape[0], 1)
    samples_max = samples_.max(dim=0)[0].repeat(samples_.shape[0], 1)
    samples_ = samples_ / samples_max
    samples_ = samples_ * (maximum - minimum) + minimum
    return samples_.reshape(s)


# def select_samples_classes(samples, labels, classes, ndims=4, rearrange_labels=True):
#     samples_list = list()
#     labels_list = list()
#     if type(classes) is int:
#         classes = list(range(classes))
#     for n, x in enumerate(classes):
#         which = (labels == x)
#         samples_list.append(samples[which])
#         if rearrange_labels:
#             labels_list.append(tc.ones(samples_list[-1].shape[0], ) * n)
#         else:
#             labels_list.append(labels[labels == x])
#     samples_ = tc.cat(samples_list, dim=0)
#     labels_ = tc.cat(labels_list, dim=0)
#     if samples_.ndimension() == 3:
#         if ndims == 4:
#             samples_ = samples_.reshape((-1, 1) + samples_.shape[1:])
#     return samples_, labels_


def split_time_series(data, length, device=None, dtype=tc.float32):
    """
    利用length长度的时序数据预测第length+1位置的数据
    :param data: 一维时序数据
    :param length: 样本长度
    :param device: 计算设备
    :param dtype: 数据精度
    :return: N * length维的样本矩阵，N维的标签向量
    """
    samples, targets = list(), list()
    device = choose_device(device)
    for n in range(length, data.numel()):
        samples.append(data[n-length:n].clone().reshape(1, -1))
        targets.append(data[n].clone())
    return tc.cat(samples, dim=0).to(
        device=device, dtype=dtype), tc.tensor(targets).to(device=device, dtype=dtype)


def split_dataset_train_test(samples, labels):
    num_c = tc.max(labels.flatten()) + 1
    train_samples, test_samples = list(), list()
    train_labels, test_labels = list(), list()
    for n in range(num_c):
        train_, test_ = split_samples(samples[labels == n])
        train_samples.append(train_)
        test_samples.append(test_)
        train_labels.append(n*tc.ones(train_.shape[0], dtype=tc.int64, device=labels.device))
        test_labels.append(n*tc.ones(test_.shape[0], dtype=tc.int64, device=labels.device))
    train_samples = tc.cat(train_samples, dim=0)
    train_labels = tc.cat(train_labels, dim=0)
    test_samples = tc.cat(test_samples, dim=0)
    test_labels = tc.cat(test_labels, dim=0)
    return train_samples, train_labels, test_samples, test_labels


def split_samples(samples, ratio=0.2, shuffle=True):
    num_train = int(samples.shape[0] * ratio)
    if shuffle:
        rand = tc.randperm(samples.shape[0])
        train_samples = samples[rand[:num_train]]
        test_samples = samples[rand[num_train:]]
    else:
        train_samples = samples[:num_train]
        test_samples = samples[num_train:]
    return train_samples, test_samples

