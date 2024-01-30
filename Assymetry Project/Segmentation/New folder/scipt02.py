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
df = pd.read_excel(os.path.join(main_dir,file_names[1]), header=1)
# Creating gaitData obejct out of the main dataframe
test_df = gaitData(df)
test_df.detrend()
# Extracting limits
cycle_lim = lim_ex(test_df)
cycle_lim_r = lim_ex_r(test_df)
# Segmentation
cycles = Cycles(test_df, cycle_lim)
cycles_r = Cycles(test_df, cycle_lim_r)

#%% Plotting
x = cycles.left_hip.sagittal
t = cycles.left_hip.time
plt.figure(figsize=(22,8))
for i in range(len(x)):
    
    plt.plot(t[i],x[i])

# Right segmentation

x = cycles_r.right_knee.sagittal
t = cycles_r.right_knee.time
plt.figure(figsize=(22,8))
for i in range(len(x)):
    
    plt.plot(t[i],x[i])

