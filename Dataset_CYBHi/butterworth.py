from scipy.signal import filtfilt
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt

'''
Change the fs 
'''

def read_signals(filepaths, FILTERED_DATA_PATH):
    """
    Read the files and and save the raw ecg values in CSV file
    :param filepaths: Path of the files
    :param RAW_SIGNAL_PATH: Base Folder where to save the files
    :return:
    """
    present=set()
    for path in filepaths:

        # 20120111-MP-A0-35.txt Take MP for example
        file_name = path.rsplit("-")[2]
        if file_name in present:
            continue
        else:
            present.add(file_name)

        data_value = np.loadtxt(path)
        data_value = data_value[:120000]


        file_path = os.path.join(FILTERED_DATA_PATH, file_name + ".csv")
        plot(data_value, file_path)


def plot(signals, FILE_PATH):
    filtered_signal=bandPassFilter(signals)
    np.savetxt(FILE_PATH, filtered_signal, fmt="%.2f")

    # Plot the graph
    # samples = 1000
    # time = np.linspace(0, 0.2, samples)
    # y1 = signals[:samples]
    # y2 = filtered_signal[:samples]
    #
    # # Plot the results
    # plt.plot(time, y1, 'r--', time, y2, 'b--')
    # plt.xlabel('time (s)')
    # plt.ylabel('Unfiltered vs Data filtered with Bandpass filter')
    # plt.show()

def bandPassFilter(signal):
    fs=1000.0
    lowcut=5
    highcut=50
    nyq=0.5*fs
    low=lowcut/nyq
    high=highcut/nyq
    order=5

    b,a=scipy.signal.butter(order, [low,high],'bandpass',analog=False)
    # b,a=scipy.signal.butter(order, low, btype='low', analog=False)
    y=scipy.signal.filtfilt(b,a,signal,axis=0)
    return y



