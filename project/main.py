from load_data import load
import plot

# ------------------------------- #
#        load training data       #
# ------------------------------- #

training_data = load('raw_weekend.csv')

# ------------------------------- #
#              plots              #
# ------------------------------- #

# plot all data
#for d in training_data:
#	plot.data(d)

# plot specific parameter data, e.g., "CDI - Distance"
for d in training_data:
	if d.name == 'CDI - Distance':
		plot.data(d)

# ------------------------------- #
#              train              #
# ------------------------------- #

# train a model for parameter "CDI - Distance"
#for d in training_data:
#	if d.name == 'CDI - Distance':
#		d.train()