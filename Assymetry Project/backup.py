import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

file = pd.read_csv('Walking_Normal_Sync.csv')

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()

file.columns

file.iplot(kind='line',x='LeftThigh_Time', y=['LeftThigh_Acc_linX','LeftThigh_Acc_linY','LeftThigh_Acc_linZ','RightThigh_Acc_linX','RightThigh_Acc_linY','RightThigh_Acc_linZ'],size=20)
file.iplot(kind='line',x='LeftThigh_Time', y=['LeftThigh_Acc_X','LeftThigh_Acc_Y','LeftThigh_Acc_Z','RightThigh_Acc_X','RightThigh_Acc_Y','RightThigh_Acc_Z'],size=20)
sns.scatterplot(data = file , x = 'RightHumerus_Time', y = 'RightShank_Time')

file2 = pd.read_csv('Walking_Normal_Sync_Joints_Kinematics.csv')
file3 = pd.read_csv('Wakling normal_Joints_Kinematics.csv')
file4 = pd.read_csv('walking normal_Joints_KinematicsB.csv')
file4.iplot(kind='line',x='LeftHip_Time', y=['LeftHip_Abduction-Adduction','LeftHip_Internal-External Rotat','LeftHip_Flexion-Extension','RightHip_Abduction-Adduction','RightHip_Internal-External Rotat','RightHip_Flexion-Extension'],size=20)
file3.iplot(kind='line',x='LeftHip_Time', y=['LeftHip_Abduction-Adduction','LeftHip_Internal-External Rotat','LeftHip_Flexion-Extension','RightHip_Abduction-Adduction','RightHip_Internal-External Rotat','RightHip_Flexion-Extension'],size=20)
file.iplot(kind='line',x='RightThigh_Time', y=['RightThigh_Acc_X','RightThigh_Acc_Y','RightThigh_Acc_Z'],size=20)
file2.iplot(kind='line',x='LeftHip_Time', y=['LeftHip_Abduction-Adduction','LeftHip_Internal-External Rotat','LeftHip_Flexion-Extension','RightHip_Abduction-Adduction','RightHip_Internal-External Rotat','RightHip_Flexion-Extension'],size=20)
file.iplot(kind='line',x='RightHumerus_Time', y=['LeftShank_Acc_GlinY','RightShank_Acc_GlinY'],size=20)
file.iplot(kind='line',x='LeftThigh_Time', y=['LeftThigh_Q0','LeftThigh_Q1','LeftThigh_Q2','LeftThigh_Q3','RightThigh_Q0','RightThigh_Q1','RightThigh_Q2','RightThigh_Q3'],size=20)

import scipy
print(scipy.__file__)
signal = file2['LeftHip_Internal-External Rotat']

import scipy
from scipy.fft import fft, fftfreq

import numpy as np
import matplotlib.pyplot as plt

# Assuming 'signal' is your signal data and 't' is the time array
signal = file2['LeftHip_Internal-External Rotat']
t = file2['LeftHip_Time']

# Compute the Fourier transform
fourier_transform = np.fft.fft(signal)

# Compute the frequencies for the Fourier transform
freq = np.fft.fftfreq(t.shape[-1])

# Plot the frequency spectrum
plt.plot(freq, np.abs(fourier_transform))
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Assuming 'signal' is your signal data and 't' is the time array
signal = file2['LeftHip_Internal-External Rotat']
t = file2['LeftHip_Time']

# Compute the Fourier transform
fourier_transform = np.fft.fft(signal)

# Compute the frequencies for the Fourier transform
freq = np.fft.fftfreq(t.shape[-1])

# Plot the frequency spectrum
plt.plot(freq, np.abs(fourier_transform))
plt.show()

# Remove frequencies from 0.009 to 0.011
mask = freq > 0.0049
filtered_fourier_transform = fourier_transform * mask

# Take the inverse Fourier transform of the resulting signal
filtered_signal = np.fft.ifft(filtered_fourier_transform)

# Create a trace for the filtered signal
trace = go.Scatter(x=t, y=filtered_signal.real)

# Create a layout
layout = go.Layout(title='Filtered Signal', xaxis=dict(title='Time'), yaxis=dict(title='Signal'))

# Create a Figure and plot it
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)

import numpy as np

# Assuming 'LeftHip_Time' is in seconds
time = file2['LeftHip_Time']

# Frequency of the sine wave
freq = 0.00007

# Generate the sine wave
sine_wave =10 * np.sin(2 * np.pi * freq * time +np.pi/2) - 10

# Add the sine wave to your plot
file2['Sine_Wave'] = sine_wave
file2.iplot(kind='line', x='LeftHip_Time', y=['LeftHip_Abduction-Adduction', 'Sine_Wave'], size=20)

#Feature Extracion

from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector
from keras import backend as K
import numpy as np

# Your data

nfeatures = 50


# Reshape your data for the LSTM layer
# Here I'm assuming your time series is one-dimensional and each sequence has 100 timesteps
# You might need to adjust this to fit your actual data
filtered_signal2 = filtered_signal.reshape((-1, 100, 1))

inputs = Input(shape=(100, 1))  # Adjust the shape parameters to match your data
encoded = LSTM(nfeatures)(inputs)  # We're encoding the input into 100 features

decoded = RepeatVector(100)(encoded)
decoded = LSTM(1, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)

sequence_autoencoder.compile(optimizer='adam', loss='mse')

sequence_autoencoder.fit(filtered_signal2, filtered_signal2, epochs=50, batch_size=128)

# Now you can use the `encoder` to transform your data into a 10-dimensional space
features = encoder.predict(filtered_signal2)

pd.DataFrame(features)

from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector
from keras import backend as K
import numpy as np

# Your data

nfeatures = 50


# Reshape your data for the LSTM layer
# Here I'm assuming your time series is one-dimensional and each sequence has 100 timesteps
# You might need to adjust this to fit your actual data
filtered_signal2 = filtered_signal.reshape((-1, 100, 1))

inputs = Input(shape=(100, 1))  # Adjust the shape parameters to match your data
encoded = LSTM(nfeatures)(inputs)  # We're encoding the input into 100 features

decoded = RepeatVector(100)(encoded)
decoded = LSTM(1, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)

sequence_autoencoder.compile(optimizer='adam', loss='mse')

sequence_autoencoder.fit(filtered_signal2, filtered_signal2, epochs=50, batch_size=128)

# Now you can use the `encoder` to transform your data into a 10-dimensional space
features = encoder.predict(filtered_signal2)

