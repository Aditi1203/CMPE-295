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
    paths = glob("data/100.atr")

    paths = [path[:-4] for path in paths]
    print("All paths", paths)

    return paths[:2]


def read_signals(paths):
    all_signals = []
    file_path=[]
    for rec in paths:
        # Read signals
        record = wf.rdsamp(rec, channels=[0])
        p_signals, fields = wf.rdsamp(rec, channels=[0])
        # print("Record", record)
        num_cols = len(record)
        num_rows = len(record[0])
        print("Shape of rows and column of record:", num_rows, num_cols)
        print("Shape of p_signal", len(p_signals), len(p_signals[0]))
        print("Value of field", fields)

        # Finding annotations: Earlier it was annotated near P-wave. Gradually, it has been shifted to R-wave
        annotation = wf.rdann(rec, "atr")
        all_beats = annotation.sample[:]
        beats = annotation.sample
        print("Value of all beats", all_beats)
        print("Length of all beats", len(all_beats))
        print("Length of beats", len(beats))
        print("--------------------------------")
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

                if j==1:
                    print("Printing first set of values for difference of annotations")
                    print("diff1", "diff2", diff1, diff2)
                    print("beats[j] - diff1: ",beats[j] - diff1)
                    print("beats[j] + diff2: ",beats[j] + diff2)

                segment_beat = p_signals[beats[j] - diff1: beats[j] + diff2, 0]
                # print("Size of segment:",len(segment_beat))

                for segment in segment_beat:
                    patient.append(segment)

        file_path.append(rec)

        print("Present Record name:", rec)

        print("Length of patient array: ", len(patient))
        segmented_arr= np.array(patient, dtype=np.float)
        print("np array", len(segmented_arr))
        print("Shape of signal", len(p_signals))

        all_signals.append(p_signals)
    return all_signals, fields, segmented_arr, file_path


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

    paths = get_paths()
    print("Path of the returned ECG recording/s:")
    print(paths)
    print("Reading the binary files using wfdb package:")
    signals, fields ,segment, file_list= read_signals(paths)
    print("Shape of all_signals",len(signals), len(signals[0]))
    print("Plotting the ECG signals")
    # plot_ecg(signals)
    print("Save to csv")
    from qrs_complex import save_to_csv, create_signal, signal_to_image, input_files, create_data_label
    save_to_csv(segment, file_list,'segment.csv')
    signal=create_signal(file_list,'segment.csv', signals, segment)
    print("length of signal after christov", len(signal),len(signal[0]))
    # print("length of signal after christov", len(signal), len(signal[0]))
    # print("length of signal after christov", len(signal), len(signal[2]))
    # print(signal)
    # normal_directory = signal_to_image(signal, "Images")

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
