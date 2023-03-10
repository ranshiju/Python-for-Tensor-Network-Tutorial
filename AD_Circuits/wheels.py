import torch as tc


def probabilities_adqc_classifier(psi, num_qc, num_class):
    s = psi.shape
    psi1 = psi.reshape(s[0], -1, 2 ** num_qc)
    psi1 = tc.einsum('nab,nac->nbc', psi1, psi1.conj())
    p = tc.zeros((s[0], num_class),
                 device=psi.device, dtype=psi1.dtype)
    for n in range(num_class):
        p[:, n] = psi1[:, n, n]
    p = tc.einsum('na,n->na', p, 1/(tc.norm(p, dim=1)+1e-10))
    return p
