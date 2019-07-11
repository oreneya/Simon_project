import pandas as pd
import property

def read_training_data(train_data):

	# read file
	a = pd.read_csv('./data/'+train_data)

	# clean nan columns
	for col in a.keys(): 
	    if pd.isnull(a[col][0]): 
	        del a[col]
	
	return a

def load(training_file):

	# choose serial number
	def obtain_serial_number_data(sn):
	    return data.loc[data['Serial Number'] == sn]

	# load data
	data = read_training_data(training_file)

	# --------------------------- #
	#   execute over parameters   #
	# --------------------------- #

	NOP = 10 # Number Of Parameters
	parameters = data['Parameter'][-NOP:]

	# list of parameter objects
	p = []

	for i in range(NOP):

		# name of parameter - string
		parameter = parameters.iloc[i]
		
		# limits of parameter - scalar
		lower_limit = data.loc[data.Parameter == parameter]['Lower Limit'].unique()
		upper_limit = data.loc[data.Parameter == parameter]['Upper Limit'].unique()

		# instantiate property/parameter
		p.append(property.Property(parameter, lower_limit, upper_limit))

		# execute over first 10 serial numbers.
		# they are the oldest, holding ~40 cycle history relative to the next 10 which have only about 15 cycles.
		for sn in data['Serial Number'].unique()[:10]:
		    
		    label = sn
		    data_sn = obtain_serial_number_data(sn)

		    x = data_sn.loc[data_sn.Parameter == parameter]['CycleCount']
		    y = data_sn.loc[data_sn.Parameter == parameter]['Measurement AVG']

		    # load sn data into its parameter
		    p[-1].serial_numbers.append(property.SerialNumber(name=label, x=x, y=y))

	return p