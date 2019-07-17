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

# plot specific part attribute data, e.g., "CDI - Distance"
#for d in training_data:
#	if d.name == 'CDI - Distance':
#		plot.data(d)

# ------------------------------- #
#              train              #
# ------------------------------- #

# Cross validation is necessary because of very few data samples - currently only 10. Thus I'm implementing here leave-1-out.

# Number Of Serial-Numbers
NOSN = len(training_data[0].serial_numbers) # [0] is technical - no meaning

lin = 10 # length of time-series as input
lout = 8 # length of output vector

# leave-1-out CV
attributes = []
models = []

for i in range(NOSN):

	# save 1 serial number for validation
	sn_valid_idx = i
	sn_train_idx = np.delete(np.arange(NOSN), sn_valid_idx)

	# train a model for part attribute "CDI - Upper Lip Radial Thickness"
	for d in training_data:
		if d.name == 'CDI - Upper Lip Radial Thickness':
			attr, m = d.train(lin, lout, sn_train_idx, sn_valid_idx)
			attributes.append(attr)
			models.append(m)




