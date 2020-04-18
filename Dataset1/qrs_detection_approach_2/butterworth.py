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
        signals = signals[:300000]
        plot(signals, FILE_PATH)

def plot(signals, FILE_PATH):

    filtered_signal=bandPassFilter(signals)
    np.savetxt(FILE_PATH, filtered_signal, fmt='%2.4e')
    # Plot the graph
    # samples = 1000
    # time = np.linspace(0, 0.2, samples)
    # y1 = signals[:samples]
    # y2 = filtered_signal[:samples]

    # Plot the results
    # plt.plot(time, y1, 'r--', time, y2, 'b--')
    # plt.xlabel('time (s)')
    # plt.ylabel('Unfiltered vs Data filtered with Bandpass filter')
    # plt.show()

    # Solution 2:
    # time = np.linspace(0, 0.2, 3000)
    # y1 = signals[:3000]
    # y2 = ekf[:3000]
    # y3 = eks[:3000]

    # x1 = np.linspace(0.0, 5.0)
    # x2 = np.linspace(0.0, 2.0)

    # y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
    # y2 = np.cos(2 * np.pi * x2)

    # plt.subplot(2, 1, 1)
    # plt.plot(time, y1, '.-')
    # plt.title('A tale of 2 subplots')
    # plt.ylabel('Damped oscillation')
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(time, y2, '.-')
    # plt.xlabel('time (s)')
    # plt.ylabel('Undamped')
    #
    # plt.show()

    # plt.subplot(2, 1, 1)
    # plt.plot(time, y1, 'o-')
    # plt.title('Unfiltered vs Extended Kalman Filter vs EF Smoothening')
    # plt.ylabel('Unfiltered')
    #
    # plt.subplot(2, 2, 1)
    # plt.plot(time, y1, 'o-')
    # plt.title('Unfiltered vs Extended Kalman Filter vs EF Smoothening')
    # plt.ylabel('Unfiltered')


def bandPassFilter(signal):

    fs=360.0
    lowcut=5
    highcut=40
    nyq=0.5*fs
    low=lowcut/nyq
    high=highcut/nyq

    order=5

    b,a=scipy.signal.butter(order, [low,high],'bandpass',analog=False)
    # b,a=scipy.signal.butter(order, low, btype='low', analog=False)
    y=scipy.signal.filtfilt(b,a,signal,axis=0)
    return y


# timing
#