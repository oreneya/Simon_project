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

print('Training feed-forward net')
bp = Backprop(1, 1, 20)
inputs = xorseq(1000)
targets = shift(inputs)
bp.train(inputs, targets, niter=100, eta=0.5, report=10)

print('\nTraining SRN')
srn = SRN(1, 1, 20)
srn.train(inputs, targets, niter=100, eta=0.5, report=10)
