# part 2 - input/output mechanism
import numpy as np

# read data
file_name = 'digits_train.txt'
print("Loading patterns from %s..." %file_name)
lines = []
with open('digits_train.txt', 'r') as f_train:
    lines = f_train.readlines()

pics_in = np.array([])
out_vectors = np.array([])

NOS = 2500 # number of samples
for i in range(NOS):
    out_vector = np.array(lines[15::16][i].split()).astype('int')
    out_vectors = np.append(out_vectors, out_vector)
    for j in range(1,15):
        pics_in = np.append(pics_in, np.array(lines[16*i+j].split()).astype('float'))

out_vectors = out_vectors.reshape(NOS, 10)
pics_in = pics_in.reshape(NOS, 14*14)

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



sample_num = -1
print("Sample # %d" %sample_num)
pic = Visualize(pics_in[sample_num], out_vectors[sample_num])
pic.printout()
