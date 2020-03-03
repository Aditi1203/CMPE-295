import wfdb as wf
import numpy as np
from glob import glob
import random
from matplotlib import pyplot as plt

def get_paths():
    paths = glob("mit-bih-data/*.atr")
    paths = [path[:-4] for path in paths]
    # print('Only paths', paths)
    return paths[:2]

def read_signals(paths):
    all_signals = []
    for rec in paths:
        record = wf.rdsamp(rec, channels=[0])
        p_signals, fields = wf.rdsamp(rec, channels=[0])
        num_cols = len(record)
        num_rows = len(record[0])
        annotation = wf.rdann(rec, "atr")
        all_beats = annotation.sample[:]
        beats = annotation.sample
        
        patient = []
        for i in all_beats:
            beats = list(beats)
            j = beats.index(i)
            if j!=0 and j!=(len(beats)-1):
                x = beats[j-1]
                y = beats[j+1]
                diff1 = abs(x - beats[j])
                diff2 = abs(y - beats[j])
                a = p_signals[beats[j] - diff1: beats[j] + diff2, 0]

                for k in a:
                    patient.append(k)
        
        # logging
        # print('record: ',rec)
        # print('patient record length: ',len(rec))
        segmented_arr = np.array(patient, dtype=np.float)
        # print('patient record as np array: \n',segmented_arr)
        all_signals.append(p_signals)
    return all_signals, fields, segmented_arr

def plot(signals):
    for signal in signals:
        chid = 0
        data = signal
        channel = data[:, chid]
        total = 600
        
        # sampling frequency
        fs = 360

        times = np.arange(total, dtype ="float") / fs
        plt.plot(times, channel[:total])
        plt.xlabel("Time[s]")
        plt.show()




paths = get_paths()
try:
    signals, fields, segment = read_signals(paths)
    from helper import save_to_csv, create_signal, signal_to_image, input_files, create_data_label
    save_to_csv(segment, "mit_data_segmented.csv")
except FileNotFoundError:
    print(".hea file not found")

# plot(signals)


# signal = create_signal("mit_data_segmented.csv", signals, segment)
# normal_directory = signal_to_image(signal, "mit_signal_images")

