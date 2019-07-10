import pandas as pd

def load(data):

	# read file
	a = pd.read_csv(data)

	# clean nan columns
	for col in a.keys(): 
	    if pd.isnull(a[col][0]): 
	        del a[col]
	
	return a