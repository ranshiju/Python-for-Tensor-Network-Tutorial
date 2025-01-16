"""

    print('自动微分得到的关于bias梯度为：')
    print(net.bias.grad.item())

    grad_b = 2 * (out.data - labels).mean().item()
    print('由链式公式计算得到的关于bias梯度为：')
    print(grad_b)

"""