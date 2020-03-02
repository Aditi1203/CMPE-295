import wfdb as wf
import numpy as np
from glob import glob
import random
from matplotlib import pyplot as plt


def get_paths():
    """ Get paths for data in data/mit/ directory """
    # Download if doesn't exist
    # There are 3 files for each record
    # *.atr is one of them
    # 117 is faulty one, 100 is good one
    paths = glob("data/*.atr")

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
        all_beats = annotation.sample[:]
        beats = annotation.sample
        print("all beats", all_beats)
        print("all beats", len(all_beats))
        print("beats", len(beats))
        # print("indexing", record[0][13])
        # print("indexing", record[0][68])
        # print("indexing", record[0][369])

        patient=[]
        for i in all_beats:
            beats=list(beats)
            j = beats.index(i)
            if j != 0 and j != (len(beats) - 1):
                x = beats[j - 1]
                y = beats[j + 1]
                diff1 = abs(x - beats[j]) // 2
                diff2 = abs(y - beats[j]) // 2
                # print("diff1", "diff2", diff1, diff2)
                # print(beats[j] - diff1)
                # print(beats[j] + diff2)

                a = p_signals[beats[j] - diff1: beats[j] + diff2, 0]
                # print(a)

                for k in a:
                    patient.append(k)

        print(rec)

        print("len", len(patient))
        segmented_arr= np.array(patient, dtype=np.float)
        print("np array", len(segmented_arr))
        print("Shape of signal", len(p_signals))

        all_signals.append(p_signals)
    return all_signals, fields, segmented_arr


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
    signals, fields ,segment= read_signals(paths)
    print(signals)
    print("Plotting the ECG signals")
    plot_ecg(signals)
    print("Save to csv")
    from qrs_complex import save_to_csv, create_signal, signal_to_image, input_files, create_data_label
    save_to_csv(segment, 'segment.csv')
    signal=create_signal('segment.csv', signals, segment)
    print(signal)
    normal_directory = signal_to_image(signal, "Images")

    # input_files("Images")
    # x_train, y_train = create_data_label('training_files.txt')
    # x_test, y_test = create_data_label('testing_files.txt')
    # print(x_train)
    # print(y_train)
    # print(x_test)


    # from qrs_detection import qrs_detect
    #
    # indexes, all = qrs_detect(signals, fields)
    # print("indexes", indexes)
    # print("all", all)
