import numpy as np
from backprop import Backprop
from srn import SRN

def xorseq(n):
        a = np.random.randint(2, size=n)
        b = np.random.randint(2, size=n)
        c = np.append(a,b)
        d = c.reshape(2,n).T
        e = np.empty(3*n)
        for i in range(len(d)):
                e[3*i : 3*(i+1)] = a[i], b[i], a[i] ^ b[i]
        e = e.reshape(3*n, 1)
        return e.astype('int')

def shift(x):
        shifted = np.append(x[1:], 0)
        return shifted.reshape(len(x), 1)

def xorerr(pred, targ):
	err = pred - targ
	return err**2

print('\nTraining feed-forward net')
bp = Backprop(1, 1, 20)
inputs = xorseq(1000)
targets = shift(inputs)
bp.train(inputs, targets, niter=100, eta=0.5, report=10)

print('\nTraining SRN')
srn = SRN(1, 1, 20)
srn.train(inputs, targets, niter=100, eta=0.5, report=10)

# evaluating bp and srn on xor problem

bp_sumerr2 = 0
srn_sumerr2 = 0

for i, t in zip(inputs[1::3], targets[1::3]):

	bp_xor_pred = bp.test(i)
	bp_sumerr2 += xorerr(bp_xor_pred, t)

	srn_xor_pred = srn.test(i)
	srn_sumerr2 += xorerr(srn_xor_pred, t)

bp_xor_rms = np.sqrt(bp_sumerr2 / (len(inputs)/3))
srn_xor_rms = np.sqrt(srn_sumerr2 / (len(inputs)/3))

print("\nXOR Error:   BP = {}   SRN = {}".format(bp_xor_rms, srn_xor_rms))