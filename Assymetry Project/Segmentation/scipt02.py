import os
os.chdir(r'G:\Masters\Term 1\Motor Control\Project\DATA')
#%% Importing
from  funcs import *
import numpy as np
import matplotlib.pylab as plt
from scipy import signal
import seaborn as sns
import pandas as pd
#%% Initiation
main_dir = r'G:\Masters\Term 1\Motor Control\Project\DATA\F.M\ALL EXCEL'
file_names = os.listdir(main_dir)
# Reading a data
df = pd.read_excel(os.path.join(main_dir,file_names[4]), header=1)
# Creating gaitData obejct out of the main dataframe
test_df = gaitData(df)
test_df.detrend()
# Extracting limits
cycle_lim = lim_ex(test_df)
# Segmentation
cycles = Cycles(test_df, cycle_lim)

#%% Plotting
x = cycles.right_knee.sagittal
t = cycles.right_knee.time
plt.figure(figsize=(22,8))
for i in range(len(x)):
    
    plt.plot(t[i],x[i])





