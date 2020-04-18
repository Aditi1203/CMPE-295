import os
import wfdb as wf
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

RAW_SIGNAL_PATH='RAW_ECG_CSV'
FILTERED_DATA_PATH='BANDPASS_LP5_HP_40'

def get_paths():
    paths = glob('data1/*.atr')  # list of all the data files.

    # Get rid of the extension and sort in increasing order.
    paths = [path[:-4] for path in paths]
    paths.sort()
    return paths

# Read Signals and generate csvs of raw ecg
def read_signals(paths, RAW_SIGNAL_PATH):
    # record=0
    for rec in paths:
        record_no = rec.rsplit("/", -1)[1]
        FILE_PATH=os.path.join(RAW_SIGNAL_PATH, record_no+'.csv')
        print("here", FILE_PATH)
        record = wf.rdsamp(rec, channels=[0])
        signals = np.asarray(record[0]).flatten()
        signals=signals[:300000]
        # print("Length",signals.shape)
        np.savetxt(FILE_PATH, signals, fmt='%2.4e')
    return signals

if __name__ == "__main__":

    paths = get_paths()
    print("Path of the returned ECG recording/s:")
    # print(paths)
    print("Reading and saving binary files using wfdb package to csv:")

    if os.path.exists(RAW_SIGNAL_PATH):
        print('The directory exists')
    else:
        print("Creating directory")
        os.makedirs(RAW_SIGNAL_PATH)
    signals = read_signals(paths, RAW_SIGNAL_PATH)

    # Saving filtered data using butterworth
    if os.path.exists(FILTERED_DATA_PATH):
        print('The directory exists')
    else:
        print("Creating directory")
        os.makedirs(FILTERED_DATA_PATH)

    from butterworth import read_signals
    read_signals(paths, FILTERED_DATA_PATH)

    # print(signals, type(signals))

    # signals=np.asarray(signals[0]).flatten()
    # print(type(signals), signals.shape)
    #

    # np.savetxt('test_input.csv', signals, fmt='%2.4e')




