import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

def data(property):
	
	for sn in property.serial_numbers:
		plt.plot(sn.x, sn.y, 'o-', label=sn.name)

	plt.plot(plt.xlim(), [property.lower_limit, property.lower_limit], 'r--')
	plt.plot(plt.xlim(), [property.upper_limit, property.upper_limit], 'r--')

	plt.xlabel('CycleCount', fontsize=15)
	plt.ylabel('Measurement AVG', fontsize=15)

	plt.title(property.name)
	plt.legend(numpoints=1, loc='best')
	plt.tick_params(axis='both', labelsize=15)
	plt.grid(color='lightgray', linestyle='--')

	plt.show()