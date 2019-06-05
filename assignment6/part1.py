import numpy as np
from backprop import Backprop

# inputs & targets
inp = np.array([[0,0],[0,1],[1,0],[1,1]])
tar = np.array([[0],[1],[1],[0]])

# create & train an instance of 'XOR' neural network
pXOR = Backprop(2,1, 30)
pXOR.train(inp, tar, niter=10000, eta=10.)

def xorseq(n):
	a = np.random.randint(2, size=n)
	b = np.random.randint(2, size=n)
	c = np.append(a,b)
	d = c.reshape(2,n).T
	e = np.empty(3*n)
	for i in range(len(d)):
		e[3*i : 3*(i+1)] = a[i], b[i], np.round(pXOR.test(d[i]))
	return e.astype('int')

def shift(x):
	return np.append(x[1:], 0)

'''
# create & train an instance of 'XOR' neural network
pXOR = Backprop(2,1, 30)
pXOR.train(inp, tar, niter=10000, eta=10.)

# print 'XOR' results
print("\nTesting 'XOR' backpropagation NN:")
#[print("pattern: " + str(i) + " | result: " + str(pXOR.test(i))) for i in inp]
for i in inp:
	print("pattern: " + str(i) + " | result: " + str(pXOR.test(i)))
'''