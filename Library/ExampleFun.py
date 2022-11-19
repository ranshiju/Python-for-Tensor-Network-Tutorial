import torch as tc


def eigs_power(mat, v0=None, which='la', tau=0.01, it_time=2000, tol=1e-14):
    """
    :param mat: 输入矩阵（实对称阵）
    :param v0: 初始化向量，默认值为随机向量
    :param which: 计算哪个本征值与本征向量，
                  'la'为代数最大，'sa'为代数最小，'lm'为模最大，'sm'为模最小
    :param tau: 小的正实数，用以构造投影矩阵
    :param it_time: 最大迭代步数
    :param tol: 收敛阈值
    :return -tc.log(lm)/tau: 代数最大（tau>0）或最小（tau<0）的本征值
    :return v1: 对应的本征向量
    """
    # 初始化向量
    if v0 is None:
        v0 = tc.randn(mat.shape[1], dtype=mat.dtype)
        v0 /= v0.norm()
    v1 = v0.clone()

    # 根据which给出投影矩阵
    tau = abs(tau)
    if which.lower() == 'la':
        rho = tc.matrix_exp(tau * mat)
    elif which.lower() == 'sa':
        rho = tc.matrix_exp(-tau * mat)
    elif which.lower() == 'lm':
        rho = tc.matrix_exp(tau * (tc.matrix_power(mat, 2)))
    else:  # which.lower() == 'sm'
        rho = tc.matrix_exp(-tau * (tc.matrix_power(mat, 2)))

    lm = 1
    for n in range(it_time):  # 开始循环迭代
        v1 = rho.matmul(v0)  # 计算v1 = rho V0
        lm = v1.norm()  # 求本征值
        v1 /= lm  # 归一化v1
        # 判断收敛
        conv = (v1 - v0).norm()
        if conv < tol:
            break
        else:
            v0 = v1.clone()

    # 修正平方带来的符号丢失
    v1 = mat.matmul(v0)
    sign = tc.sign(v0.dot(v1))

    if which.lower() == 'la':
        return tc.log(lm)/tau, v1/v1.norm()
    elif which.lower() == 'sa':
        return -tc.log(lm)/tau, v1/v1.norm()
    elif which.lower() == 'lm':
        return sign * tc.sqrt(tc.log(lm)/tau), v1/v1.norm()
    else:  # which.lower() == 'sm'
        return sign * tc.sqrt(-tc.log(lm)/tau), v1/v1.norm()