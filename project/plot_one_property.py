import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import pandas as pd

#a = pd.read_csv('raw_cleaned.csv')
a = pd.read_csv('raw_weekend.csv')

# cleaning nan columns
for col in a.keys(): 
    if pd.isnull(a[col][0]): 
        del a[col]

# choose serial number
def SN(sn):
    return a.loc[a['Serial Number'] == sn]#'SN{:02d}'.format(n)]

aa = SN('SN01')

# choose parameter
#parameters = a['Parameter'][-10:]
#parameter = parameters[0]
parameter = 'CDI - Avg Wall Cell Thickness'
#parameter = 'CDI - Distance'
lower_limit = aa.loc[aa.Parameter == parameter]['Lower Limit']
upper_limit = aa.loc[aa.Parameter == parameter]['Upper Limit']

for sn in a['Serial Number'].unique():
    label = sn
    aa = SN(sn)
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
