import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import pandas as pd
from load_data import load
import plot

# load data
data = load('raw_weekend.csv')

# choose serial number
def serial_number(sn):
    return data.loc[data['Serial Number'] == sn]

# execute over 10 parameters
parameters = data['Parameter'][-10:]

for i in range(10):

	# name of parameter - string
	parameter = parameters.iloc[i]
	
	# limits of parameter - scalar
	lower_limit = data.loc[data.Parameter == parameter]['Lower Limit'].unique()
	upper_limit = data.loc[data.Parameter == parameter]['Upper Limit'].unique()

	# execute over first 10 serial numbers.
	# they are the oldest, holding ~40 cycle history relative to the next 10 which have only about 15 cycles.
	for sn in data['Serial Number'].unique()[:10]:
	    
	    label = sn
	    data_sn = serial_number(sn)

	    x = data_sn.loc[data_sn.Parameter == parameter]['CycleCount']
	    y = data_sn.loc[data_sn.Parameter == parameter]['Measurement AVG']

	    plt.plot(x,y, 'o-', label=label)

	# plot all collected serial numbers per parameter
	plot.visuals(parameter, lower_limit, upper_limit)