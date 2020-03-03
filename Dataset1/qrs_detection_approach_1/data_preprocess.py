import wfdb as wf
import numpy as np
from glob import glob
import random
from matplotlib import pyplot as plt


def get_paths():
    """ Get paths for data1 in data1/mit/ directory """
    # Download if doesn't exist
    # There are 3 files for each record
    # *.atr is one of them
    # 117 is faulty one, 100 is good one
    paths = glob("data1/*.atr")

    paths = [path[:-4] for path in paths]
    print("Only path", paths)

    return paths[:2]


def read_signals(paths):
    all_signals = []
    for rec in paths:
        # Read signals
        record = wf.rdsamp(rec, channels=[0])
        p_signals, fields = wf.rdsamp(rec, channels=[0])
        print("Record", record)
        num_cols = len(record)
        num_rows = len(record[0])
        print("No of rows and column", num_rows, num_cols)
        print("Shape of signal", p_signals)
        annotation = wf.rdann(rec, "atr")
        print(annotation)
        all_signals.append(p_signals)
    return all_signals, fields


def plot_ecg(signals):

    for signal in signals:
        chid = 0
        data = signal
        channel = data[:, chid]
        print(channel)

        total = 600
        # Sampling frequency
        fs = 360
        # Calculate time values in seconds
        times = np.arange(total, dtype="float") / fs
        plt.plot(times, channel[:total])
        plt.xlabel("Time [s]")
        plt.show()


if __name__ == "__main__":
    print("Path of the ECG recording:")
    paths = get_paths()
    print("Reading the binary files:")
    signals, fields = read_signals(paths)
    print(signals)
    print("Plotting the ECG signals")
    # plot_ecg(signals)

    from qrs_detection import qrs_detect, qrs_segment
    from encoding import one_hot_encoding

    indexes, all_signals = qrs_detect(signals, fields, 2)
    print("--------indexes-----", indexes)
    print("all", all_signals)


    for qrs, signal, record in zip(indexes, all_signals, paths):
        segments = qrs_segment(indexes, signal)
        one_hot_encoding(segments, 'record' + record[5:])  # [5:] will get rid of the 'data1' prefix in the path.
        # for segment in segments:
        #     print(segment)
        #     print(" ")