import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from obspy.core import *

#1. Read file
#Path to folder earthquake database
e_file='earthquakes_db/524_NNA_N.mseed' #example to read 1 file
st=read(e_file)
st.plot()

#2. Signal limit
init_time=st[0].stats.starttime
#st.trim(init_time+30, init_time+150)
#st.plot()

#3. Setting N samples, Fs
N=st[0].stats.npts
Fs=st[0].stats.sampling_rate

#4. Filter and removing the mean
st.filter("bandpass", freqmin=1.0, freqmax=15.0, zerophase=True, corners=128)
#st[0].data=st[0].data-np.mean(st[0].data)
#st.filter("highpass", freq=1.0, zerophase=True, corners=128)
st.plot()

#st[0].data is the numpy stream of the digitizer data

#5 Spectral analysis
N_fft=512
f, Pxx= signal.welch(st[0].data, Fs, window='hamming', nperseg=N_fft, noverlap=N_fft*75//100) #75% overlapping
print('Frequency sample',Fs)


plt.figure()
plt.plot(f, Pxx)
plt.title('Welchs PSD')
plt.grid()
plt.show()
plt.close()

#Data numpy array
print(st[0].data)








