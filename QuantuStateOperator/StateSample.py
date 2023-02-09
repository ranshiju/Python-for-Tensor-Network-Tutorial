import torch as tc
from Library.MathFun import hadamard


def qubit_state_sampling(state, num_sample=1000, counter=True):
    """
    # 收录于Library.QuantumTool
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
    y = random.choices(
        population, weights=p.flatten(), k=num_sample)
    if counter:
        y = Counter(y)
    return y


psi = tc.tensor([1.0, 0.0], dtype=tc.float64)
psi1 = hadamard().matmul(psi)
print('初态 = ', list(psi.numpy()))
print('末态 = ', list(psi1.numpy()))

samples = qubit_state_sampling(psi1, num_sample=5000)
print('末态上的采样结果：\n', samples)


