import numpy as np
import matplotlib.pylab as plt
from scipy import signal
import seaborn as sns
import os
import pandas as pd
import warnings


def plot_signals(df, columns=['frontal','horizontal','sagittal'],\
                 legend=True, title='title'):

    '''
    This function takes a dataframe as an input and plots the signal of joints
    '''
    for col in columns:

        plt.plot(df['time'],df[col], label=col)
    if legend:
        plt.legend()
    plt.title(title)
    plt.show()


warnings.filterwarnings("ignore")


class gaitData():
    """
    A class for processing gait data.

    Attributes:
        data (pd.DataFrame): Raw dataframe of the data.
        right_knee (pd.DataFrame): Subset of data for right knee.
        right_hip (pd.DataFrame): Subset of data for right hip.
        right_shoulder (pd.DataFrame): Subset of data for right shoulder.
        left_knee (pd.DataFrame): Subset of data for left knee.
        left_shoulder (pd.DataFrame): Subset of data for left shoulder.
        left_hip (pd.DataFrame): Subset of data for left hip.
        left_ankle (pd.DataFrame): Subset of data for left ankle.
        pelvic (pd.DataFrame): Subset of data for pelvic movement.
        right_ankle (pd.DataFrame): Subset of data for right ankle.
        truth_ground (pd.DataFrame): Subset of data for trunk to ground movement.
        dfs (list): List of data subsets.

    Methods:
        __init__(self, data=None): Initializes the GaitData object.
        detrend(self): Detrends the data.
        copy(self): Creates a copy of the GaitData object.
    """
    def __init__ (self, data=None):
        """
        Initializes the GaitData object.

        Args:
            data (pd.DataFrame, optional): Raw dataframe of the data.
        """

        self.data = data
        self.right_knee = self.data[[       'RightKnee_Time',
                                            'RightKnee_Abduction-Adduction',
                                            'RightKnee_Internal-External Rotat',
                                            'RightKnee_Flexion-Extension'
                                    ]]
        self.right_hip = self.data[[        'RightHip_Time',
                                            'RightHip_Abduction-Adduction',
                                            'RightHip_Internal-External Rotat',
                                            'RightHip_Flexion-Extension'
                                    ]]
        self.right_shoulder = self.data[[   'RightShoulder_Time',
                                            'RightShoulder_Abduction-Adduction',
                                            'RightShoulder_Internal-External Rotat',
                                            'RightShoulder_Flexion-Extension'
                                        ]]
        self.left_knee = self.data[[        'LeftKnee_Time',
                                            'LeftKnee_Abduction-Adduction',
                                            'LeftKnee_Internal-External Rotat',
                                            'LeftKnee_Flexion-Extension'
                                    ]]
        self.left_shoulder = self.data[[    'LeftShoulder_Time',
                                            'LeftShoulder_Abduction-Adduction',
                                            'LeftShoulder_Internal-External Rotat',
                                            'LeftShoulder_Flexion-Extension'
                                        ]]
        self.left_hip = self.data[[         'LeftHip_Time',
                                            'LeftHip_Abduction-Adduction',
                                            'LeftHip_Internal-External Rotat',
                                            'LeftHip_Flexion-Extension'
                                    ]]
        self.left_ankle = self.data[[       'LeftAnkle_Time',
                                            'LeftAnkle_Abduction-Adduction',
                                            'LeftAnkle_Internal-External Rotat',
                                            'LeftAnkle_Flexion-Extension'
                                    ]]
        self.pelvic = self.data[[           'Pelvic_Time',
                                            'Pelvic_Abduction-Adduction',
                                            'Pelvic_Internal-External Rotat',
                                            'Pelvic_Flexion-Extension'
                                ]]
        self.right_ankle = self.data[[      'RightAnkle_Time',
                                            'RightAnkle_Abduction-Adduction',
                                            'RightAnkle_Internal-External Rotat',
                                            'RightAnkle_Flexion-Extension'
                                    ]]
        self.truth_ground =self.data[[      'Trunk2Ground_Time',
                                            'Trunk2Ground_Abduction-Adduction',
                                            'Trunk2Ground_Internal-External Rotat',
                                            'Trunk2Ground_Flexion-Extension'
                                    ]]

        names = ['time', 'frontal', 'horizontal', 'sagittal']
        self.dfs = [    self.right_ankle, self.right_knee, self.right_hip, self.right_shoulder,
                        self.left_ankle, self.left_knee, self.left_hip, self.left_shoulder,
                        self.pelvic, self.truth_ground      ]
        for i in self.dfs:

            i.columns = names
            i['time'] = (i['time'] - i['time'].values[0]) / 1000
        

    # Detranding all signals
    def detrend(self):
        """
        Detrends the data by removing trends from the signals.
        """
        cols=['frontal','horizontal','sagittal']
        for data in self.dfs:

            for col in cols:

                data[col] = signal.detrend(data[col].values)
    def copy(self):
        """
        Creates a copy of the GaitData object.

        Returns:
            GaitData: A copy of the GaitData object.
        """
        new_data = self.data.copy()
        return gaitData(new_data)
    
    def __getitem__(self, key):
        """
        Retrieves an attribute by key.

        Args:
            key (str): Attribute name.

        Returns:
            pd.DataFrame: The attribute corresponding to the key.
        """
        return getattr(self, key)



class Cycles():
    '''
    A class for extracting gait cycles
    
    Attributes:
        data : pre-processed data(gaitData)
        limits : limits of all gait cycles
        right_ankle : Cycles for right ankle
        right_knee : Cycles for right knee
        right_hip : Cycles for right hip
        right_shoulder : Cycles for right shoulder
        left_ankle : Cycles for left ankle
        left_knee : Cycles for left knee
        left_hip : Cycles for left hip
        left_shoulder : Cycles for left shoulder
        truth_ground : Cycles for truth ground
        pelvic : Cycles for pelvic
    
    Methods:
        __init__(self, data=None): Initializes the Cycles object.
        
    '''
    def __init__ (self,data,limits):
        '''
        Initialize the Cycles object.
        
        Parameters:
            data : pre-processed data(gaitData)
                The pre-processed gait data used for extracting gait cycles.
            limits : limits of all gait cycles
                The limits of all gait cycles.
                
        '''
        
        self.data = data
        self.limits = limits
        joints = ['right_ankle','right_knee', 'right_hip','right_shoulder',
                  'left_ankle', 'left_knee', 'left_hip', 'left_shoulder',
                  'truth_ground','pelvic']
        self.right_ankle = cycle_df(data.right_ankle,limits)
        self.right_knee = cycle_df(data.right_knee,limits)
        self.right_hip = cycle_df(data.right_hip,limits)
        self.right_shoulder = cycle_df(data.right_shoulder,limits)
        self.left_ankle = cycle_df(data.left_ankle,limits)
        self.left_knee = cycle_df(data.left_knee,limits)
        self.left_hip = cycle_df(data.left_hip,limits)
        self.left_shoulder = cycle_df(data.left_shoulder,limits)
        self.truth_ground = cycle_df(data.truth_ground,limits)
        self.pelvic = cycle_df(data.pelvic,limits)

def read_data(dir):
    r'''
    Reading the excel file of the data the right way
    '''

    df = pd.read_excel(dir, header=1)
    return df



def main_freq(sig, plot=False):

    # Fourier transform
    FFT =np.fft.fft(sig)
    freqs = np.fft.fftfreq(len(sig))
    period = 1.0/freqs
    if plot:

        plt.plot(abs(period),abs(FFT)/len(freqs)) 
        plt.show()
    frequencies = abs(FFT)
    i_maxfreq = np.argmax (frequencies)
    main_frequency = abs(freqs)[i_maxfreq]

    return main_frequency, (abs(freqs),abs(FFT)/len(freqs))





def signal_transform(sig):

    global FFT, new_Xph, FFT_abs
    # Fourier transfoerm of the signal
    FFT =np.fft.fft(sig)
    # Taking the length of the transformed data
        # We divide it in 2 halfs becase it is a symmetric array 
        # and we have to eliminate one of the halfs
    new_N=int(len(FFT)/2)
    # Setting the natural frequency to 1 ????????
    f_nat=1
    # It creates the frequencis array for the x axis
    new_X = np.linspace(10**-12, f_nat/2, new_N, endpoint=True)
    # Turning frequency into T
    new_Xph=1.0/(new_X)
    # Getting the absolute array of FFT
    FFT_abs=np.abs(FFT)
    # Normalizing the frequency array
    fereqance = 2*FFT_abs[0:int(len(FFT)/2.)]/len(new_Xph)

    return new_Xph, fereqance, FFT_abs

#Defining the amplitude filtering function
def fft_filter_amp(th):
    fft_tof=FFT.copy()
    fft_tof_abs=np.abs(fft_tof)
    fft_tof_abs=2*fft_tof_abs/len(new_Xph)
    fft_tof_abs[fft_tof_abs<=th]=0
    return fft_tof_abs[0:int(len(fft_tof_abs)/2.)]

# Returns an array of filtered signal
def signal_filter (perc):
    th=perc*(2*FFT_abs[0:int(len(FFT)/2.)]/len(new_Xph)).max()
    fft_tof=FFT.copy()
    fft_tof_abs=np.abs(fft_tof)
    fft_tof_abs=2*fft_tof_abs/len(new_Xph)
    fft_tof[fft_tof_abs<=th]=0
    fft_tof = np.fft.ifft(fft_tof)
    return fft_tof

def cycle_index(filtered_signal):

    i_roots = []
    # Loop to find root indecies
    for i in range(len(filtered_signal)-2):

        a1 = filtered_signal[i]
        a2 = filtered_signal[i+1]
        a3 = filtered_signal[i+2]
        if a2>a1 and a2>a3:

            i_roots.append(i+1)
    cycle_indices = []
    # Create (start, end) tuples
    for i in range(len(i_roots)-1):

        a1 = i_roots[i]
        a2 = i_roots[i+1]
        cycle_indices.append((a1, a2-1))
    return cycle_indices

def signal_smooth(signal, ws=50):
    ''''
    This function will smooth the given signal with a reasonable sliding window
    '''
    smoothed =np.convolve(signal,np.ones(ws)/ws, mode='same')
    return smoothed,ws

def cycle_segment(signal,limits):
    
    cycles = dict([])
    for i in range(len(limits)):
        
        lim = limits[i]
        cycles[i] = signal[lim[0]:lim[1]]
    return cycles


def cycle_df(data,limits):

       angles = ['time', 'frontal', 'horizontal', 'sagittal']
       angle_dict = dict([])
       for angle in angles:
           
           signal = data[angle]
           cycles = cycle_segment(signal,limits)
           angle_dict[angle] = cycles
       data_frame = pd.DataFrame(angle_dict)
       return data_frame
   
def lim_ex(data, ws=50):
    
    y = data.left_hip.sagittal
    y_s, _ = signal_smooth(y,ws)
    cycle_lim = cycle_index(y_s)
    return cycle_lim

