import numpy as np
from backprop import Backprop

class Readata:
	def __init__(self, file_name, NOS=2500):
		print("\nLoading patterns from %s..." %file_name)
		lines = []
		with open(file_name, 'r') as f_data:
			lines = f_data.readlines()
		self.pics_in = np.array([])
		self.out_vectors = np.array([])
		for i in range(NOS):
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

# just visualize an one data point as example
#sample_num = 750
#print("Sample # %d" %sample_num)
#pic = Visualize(pics_in[sample_num], out_vectors[sample_num])
#pic.printout()

# create & train an instance of '2 not-2' neural network
train_data = Readata("digits_train.txt")
train_pics = train_data.pics_in
train_targets = train_data.out_vectors
train_targets = np.argmax(train_targets, axis=1) == 2
inst2not2 = Backprop(196, 1, 20)
inst2not2.train(train_pics, train_targets, niter=1000, eta=10.)
inst2not2.save(flname="part2.wgt")

# create & test an instance of '2 not-2' neural network
inst2not2 = Backprop(196, 1, 20)
inst2not2.load(flname="part2.wgt")
test_data = Readata("digits_test.txt")
test_pics = test_data.pics_in
test_targets = test_data.out_vectors
test_targets = np.argmax(test_targets, axis=1) == 2
res = []
for i in range(len(test_pics)):
	res.append(inst2not2.test(test_pics[i]))
res = np.array(res).ravel()

# success rates
tp = sum((res > 0.5) * test_targets) / float(len(res)) # true positive rate
tn = sum(((res > 0.5) + test_targets) == 0) / float(len(res)) # true negative rate
fp = sum(((res > 0.5) - test_targets) > 0) / float(len(res)) # false positive rate
fn = sum(((res > 0.5) - test_targets) < 0) / float(len(res)) # false negative rate

# print summary
print("Part 2 - summary of results\n")
print("rate of true positives: %.3f" %tp)
print("rate of true negatives: %.3f" %tn)
print("rate of false positives: %.3f" %fp)
print("rate of false negatives: %.3f" %fn)
print("accuracy: %.3f" %(tp+tn))

