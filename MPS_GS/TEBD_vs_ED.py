from Algorithms.PhysAlgoTensor import ED_spin_chain
from Algorithms.MPS_algo import tebd_spin_chain
from Library.BasicFun import plot


print('Simulate the ground state of Heisenberg chain:')
print('Set the parameters of ED...')
para = {
    'length': 10,
    'jxy': 1,
    'jz': 1
}
E0_ed = ED_spin_chain(para=para)[0][0] / para['length']
print('ED: E0 per site = %g' % E0_ed)

print('Set the parameters of TEBD')
chi = [2, 3, 4, 6, 8, 10, 12, 14, 16]
E0_tebd = list()

para['print'] = False
for chi_now in chi:
    print('For chi = %i' % chi_now)
    paraMPS = {
        'chi': chi_now,
        'length': para['length']
    }
    out = tebd_spin_chain(
        para=para, paraMPS=paraMPS, output=['eb'])
    E0_tebd.append(out['eb'].sum().item() / paraMPS['length'])
    print('E0 per site = %g (relative error = %g)' % (
        E0_tebd[-1], abs((E0_tebd[-1] - E0_ed) / E0_ed)))

plot(chi, [E0_ed.item()] * len(chi), E0_tebd, marker=['', 's'],
     linestyle=['--', '-'], xlabel='chi', ylabel='E0',
     legend=['ED', 'TEBD'])


