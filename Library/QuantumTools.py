import torch as tc
import random
import collections
import Library.BasicFun as bf


def qubit_state_sampling(state, num_sample=1000, counter=True):
    """
    :param state: quantum state
    :param num_sample: number of samples
    :param counter: whether counter
    :return: sampling results
    """
    p = state * state.conj()
    population = bf.binary_strings(state.numel())
    y = random.choices(population, weights=p.flatten(), k=num_sample)
    if counter:
        y = collections.Counter(y)
    return y


