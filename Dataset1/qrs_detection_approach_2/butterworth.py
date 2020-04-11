from scipy.signal import filtfilt
import wfdb as wf
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt

def read_signals(paths, FILTERED_DATA_PATH):
    # record=0
    for rec in paths:
        record_no = rec.rsplit("/", -1)[1]
        FILE_PATH=os.path.join(FILTERED_DATA_PATH, record_no+'.csv')
        record = wf.rdsamp(rec, channels=[0])
        signals = np.asarray(record[0]).flatten()
        plot(signals, FILE_PATH)

def plot(signals, FILE_PATH):

    filtered_signal=bandPassFilter(signals)
    np.savetxt(FILE_PATH, filtered_signal, fmt='%2.4e')
    # Plot the graph
    # samples = 40000
    # time = np.linspace(0, 0.002, samples)
    # y1 = signals[:samples]
    # y2 = filtered_signal[:samples]

    # Plot the results
    # plt.plot(time, y1, 'r--', time, y2, 'b--')
    # plt.xlabel('time (s)')
    # plt.ylabel('Unfiltered vs Data filtered with Bandpass filter')
    # plt.show()



def bandPassFilter(signal):

    fs=360.0
    lowcut=5.0
    highcut=40.0
    nyq=0.5*fs
    low=lowcut/nyq
    high=highcut/nyq

    order=4

    b,a=scipy.signal.butter(order, [low,high],'bandpass',analog=False)
    y=scipy.signal.filtfilt(b,a,signal,axis=0)
    return y


# timing
#