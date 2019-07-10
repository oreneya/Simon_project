import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

def visuals(parameter, lo_lim, up_lim):

	plt.plot(plt.xlim(), [lo_lim,lo_lim], 'r--')
	plt.plot(plt.xlim(), [up_lim,up_lim], 'r--')
	
	plt.xlabel('CycleCount', fontsize=15)
	plt.ylabel(parameter, fontsize=15)
	
	plt.title(parameter)
	plt.legend(numpoints=1, loc='best')#, loc='center left', bbox_to_anchor=(1.,0.5))
	plt.tick_params(axis='both', labelsize=15)
	plt.grid(color='lightgray', linestyle='--')
	
	plt.show()