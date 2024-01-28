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
# Defining y and t to work with
y = test_df.right_hip.sagittal
t = test_df.right_hip.time


#%% Coding
# Fourier transform of the given signal
fig1 = plt.figure(figsize=(8,6))
x03, y03, FFT_abs = signal_transform(y)
plt.plot(x03,y03)
plt.xlim((0,1000))
plt.xlabel('Period(ms)',fontsize=10)
plt.ylabel('Size', rotation=0, labelpad=15,fontsize=10)
plt.title('Fourier Transform', fontweight='bold', fontsize=16)
plt.show()
# Filtering frequencies
filter_th = 20
y04 = fft_filter_amp(filter_th)
new_Xph = x03
fig2 = plt.figure(figsize=(8,6))
plt.plot(x03,y03,label='Main frequencies', color='blue')
plt.plot(new_Xph,y04,label='Filtered frequencies', color='red')
plt.xlabel('Period(ms)',fontsize=10)
plt.ylabel('Size', rotation=0, labelpad=15,fontsize=10)
plt.title('Filtered frquencies', fontweight='bold', fontsize=16)
plt.xlim((0,5000))
plt.legend()
plt.show()
# The filtered signal
y_filtered = signal_filter(0.95)
plt.figure(figsize=(12,3))
plt.plot(t,y,label='Raw signal',color='red')
plt.plot(t,y_filtered,label='Filtered signal',color='blue')
plt.title('Filtered signal', fontweight='bold', fontsize=16)
plt.xlabel('Time(s)',fontsize=10)
plt.legend()
plt.show()
# Smoothing the signal
y_s, _ = signal_smooth(y)
fig3 = plt.figure(figsize=(22,8))
plt.plot(t,y_s,label='Smoothed signal',color='red')
plt.plot(t,y, label='Main signal',color='blue')
plt.title('Smoothed signal', fontweight='bold', fontsize=16)
plt.xlabel('Time(s)',fontsize=10)
plt.xlim([5, 10])
plt.legend()
# #cycles
cycle_lim = cycle_index(y_s)
cycle_lim1 = cycle_index(y_filtered)
print(f'\nThere are {len(cycle_lim)} cycles in the given signal')
# Cycle Segmentation
## Moving average
cycles = cycle_segment(y,cycle_lim) # Signal segmentation
time = cycle_segment(t,cycle_lim) # Time segmentation
fig4 = plt.figure(figsize=(22,8))
### Plot segmentation
for i in range(len(cycles)):
    
    plt.plot(time[i],cycles[i])
plt.xlabel('Time')
plt.ylabel('Manitude')
plt.title('Segmented signal with "Moving average"')
plt.show()
## Frequency filtering
cycles1 = cycle_segment(y,cycle_lim1) # Signal segmentation
time = cycle_segment(t,cycle_lim1) # Time segmentation
fig5 = plt.figure(figsize=(22,8))
### Plot segmentation
for i in range(len(cycles1)):
    
    plt.plot(time[i],cycles1[i])
plt.xlabel('Time')
plt.ylabel('Manitude')
plt.title('Segmented signal with "Frequency filtering"')
plt.show()

#%% Segmentation
cycles = Cycles(test_df, left_lim)








#%% TEST

# Segmentation limits

y_left = test_df.left_hip.sagittal
left_s, _ = signal_smooth(y_left,100)
left_lim = cycle_index(left_s)
# left_cycles = cycle_segment(y_left,left_lim)
# time = cycle_segment(t,left_lim)
# fig_test = plt.figure(figsize=(22,8))
# for i in range(len(left_cycles)):
    
#     plt.plot(time[i],left_cycles[i])
# plt.xlabel('Time')
# plt.ylabel('Manitude')
# plt.title('Segmented signal with "Frequency filtering"')
# plt.clf()



# # Segmenting data
# dataframe = test_df.copy()
# joints = ['right_ankle','right_knee', 'right_hip','right_shoulder',
#           'left_ankle', 'left_knee', 'left_hip', 'left_shoulder',
#           'truth_ground','pelvic']
# angles = ['time', 'frontal', 'horizontal', 'sagittal']
# for joint in joints: # All joints
    
#     angle_dict = dict([])
#     for angle in angles: # All angles
        
    
#         signal = test_df[joint][angle].values
#         cycles = cycle_segment(signal,left_lim)
#         angle_dict[angle] = cycles
    
# def cycle_df(data,limits):

#        angles = ['time', 'frontal', 'horizontal', 'sagittal']
#        angle_dict = dict([])
#        for angle in angles:
           
#            signal = data[angle]
#            cycles = cycle_segment(signal,limits)
#            angle_dict[angle] = cycles
#        data_frame = pd.DataFrame(angle_dict)
#        return data_frame
            
            

# This class saves the cycles of the given data
# class Ok():
    
#     def __init__ (self, data=None):
        
#         self.right_hip.sagittal.cycles = data.right_hip

#%% TEST2








