import numpy as np
from numpy import array
import wfdb
from wfdb import processing
from matplotlib import pyplot as plt


def qrs_detect(signals, fields, max_num=48):
    """
       qrs_indexes holds list of integer representing the location of QRS in the signals, the indices of the detected
       qrs complexes
       signals is ECG signal of recorded individuals
    """

    fs = fields["fs"]
    qrs_indexes = []
    all_signals = []
    for signal in signals[:max_num]:
        print("Signals: matrix form", signal)

        # Converting into array
        arr_signal = np.array(signal).ravel()
        print("Signals: array form", arr_signal)

        qrs_index = processing.xqrs_detect(sig=signal[:, 0], fs=fs)

        # Plot the detected complex on graph
        # plt.plot(arr_signal)
        # plt.scatter([qrs_index], [arr_signal[qrs_index]], c="r")
        # plt.show()

        qrs_indexes.append(qrs_index)
        all_signals.append(arr_signal)

        # xqrs = processing.XQRS(sig=signal[:, 0], fs=fs)
        # xqrs.detect()
        # wfdb.plot_items(
        #     signal=signal, ann_samp=[xqrs.qrs_inds], title="Using XQRS", figsize=(10, 4)
        # )
        print(len(all_signals))
    qrs_indexes = np.asarray(qrs_indexes)
    return qrs_indexes, all_signals

def qrs_segment(qrs_inds, signals):
    '''
    Need to create a list of numpy arrays, each representing its own QRS
    complex. We'll use 75% of this array to train and 25% to test for
    accuracy.
    '''
    prev_ind = 0    # Lower bound on segment
    end_ind = 0;  # Upper bound on segment
    last_ind = qrs_inds[-1] # Last index in qrs_inds. Used for edge case
    segments = []   # List of numpy arrays representing ONE patient's QRS complexes
    '''
    Segment from 0 to halfway between elements of indexes 0 and 1. Use that 'halfway' 
    value to then segment between index 1 and 2 and so on. 
    Edge case: Extracting the last segment. 
    '''
    one_behind = 0

    for ind in qrs_inds[0]:
        if ind == qrs_inds[0][0]: continue # If 'ind' is the first one in the list, skip -- Fencepost algorithm
        #Case when we just need to iterate from the last prev_ind to the end of signals
        if ind == qrs_inds[0][-1]:
            '''
            Special case where we want to just get the prev_ind to the end of the list. Fence post case.
            Take values from prev_ind to len(signals[0] - 1)
            '''
            segments.append(signals[prev_ind:len(signals) - 1])
            continue
        end_ind = ((qrs_inds[0][one_behind] + ind) // 2) - 1
        segments.append(signals[prev_ind:end_ind])
        '''
        prev_ind and end_ind are what splits each QRS complex from the next.
        Split the numpy array based on this then save as binary files.
        Example: segments.append(signals[0].split(prev_ind:end_ind)) for every patient.
        '''
        prev_ind = end_ind + 1
        one_behind = one_behind + 1
    return segments


def segment_qrs(qrs_indexes, signals):
    print("segmentation qrs")

