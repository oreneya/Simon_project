import numpy as np
from backprop import Backprop

# inputs & targets
inp = np.array([[0,0],[0,1],[1,0],[1,1]])
tar = np.array([[0],[1],[1],[0]])

# create & train an instance of 'XOR' neural network
pXOR = Backprop(2,1, 3)
pXOR.train(inp, tar, niter=10000, eta=10.)

# print 'XOR' results
print("\nTesting 'XOR' backpropagation NN:")
[print("pattern: " + str(i) + " | result: " + str(pXOR.test(i))) for i in inp]
