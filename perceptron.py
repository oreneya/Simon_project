import numpy as np

class Perceptron:
    def __init__(self, n, m):
        self.n = n # number of inputs
        self.m = m # number of outputs
        self.weights = np.random.uniform(low=-0.05, high=0.05, size=(n+1,m))
    def __str__(self):
        print("A perceptron with %d inputs and %d outputs." %(self.n, self.m))
    def test(self, I): return np.dot(np.append(I, 1), self.weights) > 0
    def train(self, I, T, niter=1000):
        I = np.hstack((I, np.ones((len(I),1))))
        for i in range(niter):
            dw = np.zeros(self.weights.shape)
            for j in range(len(I)):
                out = np.dot(I[j], self.weights) > 0
                D = T[j] - out
                dw += np.outer(I[j], D)                
            self.weights += dw / len(I)
        return self.weights    
