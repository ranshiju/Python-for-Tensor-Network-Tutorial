import torch as tc
from Library.MathFun import hadamard


def qubit_state_sampling(state, num_sample=1000, counter=True):
    """
    :param state: quantum state
    :param num_sample: number of samples
    :param counter: whether counter
    :return: sampling results
    """
    import random
    from collections import Counter
    from Library.BasicFun import binary_strings

    p = state * state.conj()
    population = binary_strings(state.numel())
    y = random.choices(population, weights=p.flatten(), k=num_sample)
    if counter:
        y = Counter(y)
    return y


psi = tc.tensor([1.0, 0.0])
psi1 = hadamard().matmul(psi)
print('Initial state = ', list(psi.numpy()))
print('Final state = ', list(psi1.numpy()))

samples = qubit_state_sampling(psi1, num_sample=5000)
print('Sampling on the final state: \n', samples)



