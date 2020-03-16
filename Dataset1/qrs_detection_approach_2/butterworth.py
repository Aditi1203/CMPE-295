from scipy.signal import filtfilt
from scipy import stats
import csv
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

def plot(signals):

    all=[]
    for order in signals:
        for o in order:
            for i in o:
                all.append(i)

    narr=np.array(all)
    time= np.linspace(0,0.002,650000)
    plt.plot(time,narr)
    # plt.show()

    filtered_signal=bandPassFilter(all)
    print("-----",filtered_signal)
    plt.plot(time,filtered_signal)
    plt.show()


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