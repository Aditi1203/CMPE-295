from scipy.signal import butter, lfilter, filtfilt
from csv import reader

ecg = []

with open('data/Person_01/rec_1.csv') as csv_file:
    csv_reader = reader(csv_file, delimiter=',')
    for row in csv_reader:
        ecg.append(float(row[0]))

fs = len(ecg)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def run():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    lowcut = 500.0
    highcut = 1250.0

    T = 0.05
    nsamples = T * fs
    t = np.linspace(0, T, fs, endpoint=False)

    plt.figure(1)
    plt.clf()
    plt.plot(t, ecg, label='Noisy signal')

    y = butter_bandpass_filter(ecg, lowcut, highcut, fs, order=4)
    plt.plot(t, y, label='Filtered signal')
    plt.xlabel('time (seconds)')
    # plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()


run()
