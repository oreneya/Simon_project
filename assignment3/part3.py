import numpy as np
from backprop import Backprop
from sys import stdout

class Readata:
	def __init__(self, file_name, NOS=2500):
		print("\nLoading patterns from %s..." %file_name)
		lines = []
		with open(file_name, 'r') as f_data:
			lines = f_data.readlines()
		self.pics_in = np.array([])
		self.out_vectors = np.array([])
		for i in range(NOS):
			# track progress
			stdout.write('\r')
			stdout.write("%-12s %.1f%%" % ('Training progress... ',100*i/NOS))
			stdout.flush()
			
			out_vector = np.array(lines[15::16][i].split()).astype('int')
			self.out_vectors = np.append(self.out_vectors, out_vector)
			for j in range(1,15):
				self.pics_in = np.append(self.pics_in, np.array(lines[16*i+j].split()).astype('float'))
		self.out_vectors = self.out_vectors.reshape(NOS, 10)
		self.pics_in = self.pics_in.reshape(NOS, 14*14)
		print("Read %d patterns" %NOS)

class Visualize:
    def __init__(self, arr, target):
        self.edge = int(len(arr)**0.5)
        self.arr_str = arr.astype('str')
        self.arr_str[np.where(arr==0)] = ' '
        self.arr_str[np.where(arr>0)] = '*'
        self.picout = self.arr_str.reshape(self.edge, self.edge)
        self.tarout = np.where(target == 1)
    def printout(self):
        print("Input:")
        for i in range(self.edge):
            print(' '.join(map(str, self.picout[i])))
        print("Target: " + str(self.tarout[0]))

'''
# create & train an instance of '2 not-2' neural network
train_data = Readata("digits_train.txt")
train_pics = train_data.pics_in
train_targets = train_data.out_vectors
instant = Backprop(196, 10, 20)
instant.train(train_pics, train_targets, niter=1000, eta=10.)
instant.save(flname="part3.wgt")
'''

# create & test an instance of '2 not-2' neural network
instant = Backprop(196, 10, 20)
instant.load(flname="part3.wgt")
test_data = Readata("digits_test.txt")
test_pics = test_data.pics_in
test_targets = test_data.out_vectors

# binarize results
res = []
for i in range(len(test_pics)):
	res.append(instant.test(test_pics[i]))
	res[i] = np.floor(res[i]/np.max(res[i]))

# compute confusion matrix
conf_mat = np.zeros((10,10))
for i in range(10):
	conf_mat[i] = np.sum(res[i*250:(i+1)*250], axis=0)

# print summary
print("Part 3 - confusion matrix\n")
print(conf_mat)



