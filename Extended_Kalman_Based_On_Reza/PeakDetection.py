import numpy as np
from math import floor


def PeakDetection(x=None, ff=None, *varargin):

    print("--------Inside Peak Detection--------")

    # x = x.flatten()
    N = len(x)
    print("Signal Length:", N)
    peaks = np.zeros(shape=(1, N)).flatten()

    th = 0.5
    rng = floor(th / ff)

    print("Absolute Max value of signal:", abs(np.amax(x)))
    print("Absolute Min value of signal:", abs(np.amin(x)))

    flag = abs(np.amax(x)) > abs(np.amin(x))
    print("Flag Value:", flag)
    indices = []

    if flag:
        # print("inside if")
        for j in range(0, N):
            if j > rng and j < N - rng:
                start_index = j - rng
                end_index = j + rng
                print("Start Index, Last index:", start_index, end_index, rng)
            elif j > rng:
                start_index = N - 2 * rng
                end_index = N
            else:
                start_index = 1
                end_index = 2 * rng
            if max(x[start_index:end_index]) == x[j]:
                peaks[j] = 1

    else:
        # print("Inside Else")
        for j in range(0, N):
            if j > rng and j < N - rng:
                start_index = j - rng
                end_index = j + rng
                # print("Start Index, Last index:", start_index, end_index, rng)
            elif j > rng:
                start_index = N - 2 * rng
                end_index = N
            else:
                start_index = 1
                end_index = 2 * rng
            # print("Start Index, Last index:", start_index,end_index, rng)
            if min(x[start_index:end_index]) == x[j]:
                peaks[j] = 1
                indices.append(j)
            # if(np.amin(x[start_index:end_index])==x[j]):
            #     peaks[j]=1;
    print("------Exit Peak Detection------")

    result = np.where(peaks == 1)
    result = np.asarray(result).flatten()
    d=np.diff(result)
    return peaks
