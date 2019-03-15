import numpy as np

# ------ #
# Part 1 #
# ------ #

# Initializations
x1 = np.array([])
x2 = np.array([])
y = np.array([])
z = np.array([])

# Read data-file
with open('assign1_data.txt', 'r') as f:
	lines = f.readlines()

for line in lines[2:]:
	tmp = line.split()
	x1 = np.append(x1, float(tmp[0]))
	x2 = np.append(x2, float(tmp[1]))
	y = np.append(y, float(tmp[2]))
	z = np.append(z, float(tmp[3]))

# Calcs.
def slope(x):
    return sum((x-np.mean(x))*(y-np.mean(y))) / sum((x-np.mean(x))**2)

m1 = slope(x1)
m2 = slope(x2)

b1 = np.mean(y) - m1 * np.mean(x1)
b2 = np.mean(y) - m2 * np.mean(x2)

# print outs
print("Part 1 results:")
print("---------------\n")

print("Solutions for x1 data:")
print("m = %.2f" %m1)
print("b = %.2f\n" %b1)

print("Solutions for x2 data:")
print("m = %.2f" %m2)
print("b = %.2f\n" %b2)

# ------ #
# Part 2 #
# ------ #

# Calcs.
def fit_model(_x1, _x2, _y):
	A = np.vstack([_x1, _x2, np.ones(len(_y))]).T
	_w1, _w2, _b = np.linalg.lstsq(A, _y)[0]
	return _w1, _w2, _b

w1, w2, b = fit_model(x1, x2, y)

# print outs
print("Part 2 results:")
print("---------------\n")

print("Full linear regression solution:")
print("w1 = %.2f" %w1)
print("w2 = %.2f" %w2)
print("b = %.2f\n" %b)

# ------ #
# Part 3 #
# ------ #

# Calcs.
_z = (w1*x1 + w2*x2 + b) > 0
result = (sum(z == _z) / len(z)) * 100 # success rate in percentage

# print outs
print("Part 3 results:")
print("---------------\n")

print("Success rate: %.2f" %result +"%\n")

# ------ #
# Part 4 #
# ------ #

print("Part 4 results:")
print("---------------\n")

# model fitted to first p data-samples
for p in [25, 50, 75]:
	print("Results for training on first %d data-samples:\n" %p)
	w1, w2, b = fit_model(x1[:p], x2[:p], y[:p])
	
	# train set
	_z = (w1*x1[:p] + w2*x2[:p] + b) > 0
	result = (sum(z[:p] == _z) / len(z[:p])) * 100
	print("Success rate on train-set: %.2f" %result +"%")
	
	# test set
	_z = (w1*x1[p:] + w2*x2[p:] + b) > 0
	result = (sum(z[p:] == _z) / len(z[p:])) * 100
	print("Success rate on test-set: %.2f" %result +"%\n")

print("Baseline test for w1=w2=b=0:")
for p in [25, 50, 75]:
	print("Results for training on first %d data-samples:\n" %p)
	w1, w2, b = fit_model(x1[:p], x2[:p], y[:p])
	
	# train set
	_z = (0*x1[:p] + 0*x2[:p] + 0) > 0
	result = (sum(z[:p] == _z) / len(z[:p])) * 100
	print("Success rate on train-set: %.2f" %result +"%")
	
	# test set
	_z = (0*x1[p:] + 0*x2[p:] + 0) > 0
	result = (sum(z[p:] == _z) / len(z[p:])) * 100
	print("Success rate on test-set: %.2f" %result +"%\n")
