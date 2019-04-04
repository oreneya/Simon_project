import numpy as np
from perceptron import Perceptron

# inputs & targets
inp = np.array([[0,0],[0,1],[1,0],[1,1]])
tarAND = np.array([[0],[0],[0],[1]])
tarOR = np.array([[0],[1],[1],[1]])

# create & train an instance of 'AND' perceptron
pAND = Perceptron(2,1)
pAND.train(inp, tarAND, 5)

# create & train an instance of 'OR' perceptron
pOR = Perceptron(2,1)
pOR.train(inp, tarOR, 5)

# print 'AND' results
print("Testing 'AND' perceptron:")
[print("pattern: " + str(i) + " | result: " + str(pAND.test(i))) for i in inp]

# print 'OR results
print("\nTesting 'OR' perceptron:")
[print("pattern: " + str(i) + " | result: " + str(pOR.test(i))) for i in inp]
