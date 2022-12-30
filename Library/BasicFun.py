import math
import os
import time

import numpy as np
import torch as tc
from matplotlib import pyplot as plt


def binary_strings(num):
    s = list()
    length = len(str(bin(num-1))[2:])
    for n in range(num):
        b = str(bin(n))[2:]
        l0 = len(b)
        if length > l0:
            b = ''.join([('0' * (length-l0)), b])
        s.append(b)
    return s


def combine_dicts(dic_def, dic_new, deep_copy=False):
    # dic_def中的重复key值将被dic_new覆盖
    import copy
    if dic_new is None:
        return dic_def
    if deep_copy:
        return dict(copy.deepcopy(dic_def), **copy.deepcopy(dic_new))
    else:
        return dict(dic_def, **dic_new)


def convert_nums_to_abc(nums, n0=0):
    s = ''
    n0 = n0 + 97
    for m in nums:
        s += chr(m + n0)
    return s


def choose_device(n=0):
    if n == 'cpu':
        return 'cpu'
    else:
        if tc.cuda.is_available():
            if n is None:
                return tc.device("cuda:0")
            elif type(n) is int:
                return tc.device("cuda:"+str(n))
            else:
                return tc.device("cuda"+str(n)[4:])
        else:
            return tc.device("cpu")


def empty_list(num, content=None):
    return [content] * num


def find_indexes_value_in_list(x, value):
    return [n for n, v in enumerate(x) if v == value]


def fprint(content, file=None, print_screen=True, append=True):
    if file is None:
        file = './record.log'
    if append:
        way = 'ab'
    else:
        way = 'wb'
    with open(file, way, buffering=0) as log:
        log.write((content + '\n').encode(encoding='utf-8'))
    if print_screen:
        print(content)


def indexes_eq2einsum_eq(indexes):
    eq = convert_nums_to_abc(indexes[0])
    for n in range(1, len(indexes)-1):
        eq += (',' + convert_nums_to_abc(indexes[n]))
    eq += ('->' + convert_nums_to_abc(indexes[-1]))
    return eq


def list_eq2einsum_eq(eq):
    # 将list表示的equation转化为einsum函数的equation
    # list中的数字不能超过25！！！
    # 例如[[0, 1], [0, 2], [1, 2]] 转为 'ab,ac->bc'
    # 例如[[0, 1], [0, 1], []] 转为 'ab,ab->'
    length = len(eq)
    eq_str = ''
    for n in range(length-1):
        tmp = [chr(m+97) for m in eq[n]]
        eq_str = eq_str + ''.join(tmp) + ','
    eq_str = eq_str[:-1] + '->'
    tmp = [chr(m+97) for m in eq[-1]]
    return eq_str + ''.join(tmp)


def load(path_file, names=None, device='cpu'):
    if os.path.isfile(path_file):
        if names is None:
            data = tc.load(path_file)
            return data
        else:
            tmp = tc.load(path_file, map_location=device)
            if type(names) is str:
                data = tmp[names]
                return data
            elif type(names) in [tuple, list]:
                nn = len(names)
                data = list(range(0, nn))
                for i in range(0, nn):
                    data[i] = tmp[names[i]]
                return tuple(data)
            else:
                return None
    else:
        return None


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot(x, *y, marker='s'):
    if type(x) is tc.Tensor:
        if x.device != 'cpu':
            x = x.cpu()
        x = x.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if len(y) > 0.5:
        for y0 in y:
            if type(y0) is tc.Tensor:
                if y0.device != 'cpu':
                    y0 = y0.cpu()
                y0 = y0.numpy()
            ax.plot(x, y0, marker=marker)
    else:
        ax.plot(x, marker=marker)
    plt.show()


def print_dict(a, keys=None, welcome='', style_sep=': ', end='\n', file=None, print_screen=True, append=True):
    express = welcome
    if keys is None:
        for n in a:
            express += n + style_sep + str(a[n]) + end
    else:
        if type(keys) is str:
            express += keys.capitalize() + style_sep + str(a[keys])
        else:
            for n in keys:
                express += n.capitalize() + style_sep + str(a[n])
                if n is not keys[-1]:
                    express += end
    fprint(express, file, print_screen, append)
    return express


def print_progress_bar(n_current, n_total, message=''):
    x1 = math.floor(n_current / n_total * 10)
    x2 = math.floor(n_current / n_total * 100) % 10
    if x1 == 10:
        message += '\t' + chr(9646) * x1
    else:
        message += '\t' + chr(9646) * x1 + str(x2) + chr(9647) * (9 - x1)
    print('\r'+message, end='')
    time.sleep(0.01)


def replace_value(x, value0, value_new):
    x_ = np.array(x)
    x_[x_ == value0] = value_new
    return list(x_)


def save(path, file, data, names):
    mkdir(path)
    tmp = dict()
    for i in range(0, len(names)):
        tmp[names[i]] = data[i]
    tc.save(tmp, os.path.join(path, file))


def search_file(path, exp):
    import re
    content = os.listdir(path)
    exp = re.compile(exp)
    result = list()
    for x in content:
        if re.match(exp, x):
            result.append(os.path.join(path, x))
    return result


def sort_list(a, order):
    return [a[i] for i in order]


# -------------------------------------
# From ZZS
def compare_iterables(a_list, b_list):
    from collections.abc import Iterable
    if isinstance(a_list, Iterable) and isinstance(b_list, Iterable):
        xx = [x for x in a_list if x in b_list]
        if len(xx) > 0:
            return True
        else:
            return False
    else:
        return False


def inverse_permutation(perm):
    # perm is a torch tensor
    if not isinstance(perm, tc.Tensor):
        perm = tc.tensor(perm)
    inv = tc.empty_like(perm)
    inv[perm] = tc.arange(perm.size(0), device=perm.device)
    return inv.tolist()
