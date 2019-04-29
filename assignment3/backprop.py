import numpy as np
import pickle
from sys import stdout

class Backprop:
    def __init__(self, n, m, h):
        self.n = n # number of inputs
        self.m = m # number of outputs
        self.h = h # number of hidden units
        self.wih = np.random.uniform(low=-0.05, high=0.05, size=(n+1,h))
        self.who = np.random.uniform(low=-0.05, high=0.05, size=(h+1,m))

    def __str__(self):
        print("A perceptron with %d inputs and %d outputs." %(self.n, self.m))

    def squash(self, V):
        self.style = 'sigmoid'
        if self.style == 'sigmoid':
            return 1 / (1+np.exp(-V))
        if self.style == 'tanh':
            return np.tanh(V)
        if self.style == 'relu':
            return np.max([V, np.zeros(len(V))], axis=0)
        
    def dsquash(self, V):
        self.style = 'sigmoid'
        if self.style == 'sigmoid':
            return self.squash(V) * (1-self.squash(V))
        if self.style == 'tanh':
            return 1 - self.squash(V)**2
        if self.style == 'relu':
            return V >= 0
    
    def test(self, I):
        H_net = np.dot(np.append(I, 1), self.wih)
        H = self.squash(H_net)
        H = np.append(H, 1)
        O_net = np.dot(H, self.who)
        O = self.squash(O_net)
        return O

    def train(self, I, T, niter=1000, eta=.5):
        I = np.hstack((I, np.ones((len(I),1))))
        for i in range(niter):
            dwih = np.zeros(self.wih.shape)
            dwho = np.zeros(self.who.shape)
            for j in range(len(I)):
                H_net = np.dot(I[j], self.wih)
                H = self.squash(H_net)
                H = np.append(H, 1)
                O_net = np.dot(H, self.who)
                O = self.squash(O_net)
                O_err = (T[j]-O) * self.dsquash(O_net)
                H_err = np.dot(O_err, self.who.T)[:-1] * self.dsquash(H_net)
                dwih += np.outer(I[j], H_err)
                dwho += np.outer(H, O_err)
            self.wih += eta * dwih / len(I)
            self.who += eta * dwho / len(I)
            # track progress
            stdout.write('\r')
            stdout.write("%-12s %.1f%%" % ('Training progress... ',100*i/niter))
            stdout.flush()
        return self.wih, self.who
    
    def save(self, flname):
        pickle.dump([self.wih, self.who], open(flname, "wb"))
    
    def load(self, flname):
        weights = pickle.load(open(flname, "rb"))
        self.wih = weights[0]
        self.who = weights[1]
