
def super_diagonal_tensor(dim, order):
    import torch as tc
    """
    :param dim: bond dimension
    :param order_g: tensor order_g
    :return: high-order_g super-orthogonal tensor
    """
    delta = tc.zeros(dim ** order, dtype=tc.float64)
    delta[0] = 1
    delta[-1] = 1
    return delta.reshape([dim] * order)


print(super_diagonal_tensor(2, 3))
