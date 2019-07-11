import numpy as np
from load_data import load
import plot

# ------------------------------- #
#        load training data       #
# ------------------------------- #

training_data = load('raw_weekend.csv')

# ------------------------------- #
#     visualize training data     #
# ------------------------------- #

# plot all data
#for d in training_data:
#	plot.data(d)

# plot specific parameter data, e.g., "CDI - Distance"
#for d in training_data:
#	if d.name == 'CDI - Distance':
#		plot.data(d)

# ------------------------------- #
#              train              #
# ------------------------------- #

# Number Of Serial Numbers
NOSN = len(training_data[0].serial_numbers) # [0] is technical - no meaning

# generate random entries of serial numbers to train 70% of data
sn_train_idx = np.random.choice(np.arange(NOSN), size=int(NOSN*0.7), replace=False)
# complementary validation indices
sn_valid_idx = np.delete(np.arange(NOSN), sn_train_idx)

# train a model for parameter "CDI - Distance"
for d in training_data:
	if d.name == 'CDI - Distance':
		d.train(sn_train_idx, sn_valid_idx)