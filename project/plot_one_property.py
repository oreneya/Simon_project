import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import pandas as pd

a = pd.read_csv('raw_cleaned.csv')

# choose serial number
def SN(n):
    return a.loc[a['Serial Number'] == 'SN'+str(n)]

aa = SN(1)

# choose parameter
parameter = 'CDI - Avg Wall Cell Thickness'
#parameter = 'CDI - Distance'
lower_limit = aa.loc[aa.Parameter == parameter]['Lower Limit']
upper_limit = aa.loc[aa.Parameter == parameter]['Upper Limit']

for i in range(1,11):
    label = 'SN'+str(i)
    aa = SN(i)
    x = aa.loc[aa.Parameter == parameter]['CycleCount']
    y = aa.loc[aa.Parameter == parameter]['Measurement AVG']
    if x.shape[0]:
        plt.plot(x,y, 'o-', label=label)

plt.plot(plt.xlim(), [lower_limit,lower_limit], 'r--')
plt.plot(plt.xlim(), [upper_limit,upper_limit], 'r--')

plt.xlabel('CycleCount', fontsize=15)
plt.ylabel(parameter, fontsize=15)

plt.title(parameter)
plt.legend(numpoints=1, loc='best')#, loc='center left', bbox_to_anchor=(1.,0.5))

plt.tick_params(axis='both', labelsize=15)
plt.grid(color='lightgray', linestyle='--')

plt.show()
