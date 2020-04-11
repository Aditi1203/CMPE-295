import os
import wfdb as wf
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    print("hello")
    path1 = 'lpfilter5_output.csv'
    path2 = 'test_input.csv'
    path3= 'bt_100.csv'
    df1 = pd.read_csv(path1, sep=',', header=None)
    df2 = pd.read_csv(path2, sep=' ', header=None)
    df3 = pd.read_csv(path3, sep=' ', header=None)

    print("Read from csv", df1.shape)
    lp_filter_output = df1.to_numpy().flatten()
    lp_filter_output = np.transpose(lp_filter_output)
    print("lp filter output:", type(lp_filter_output), lp_filter_output.shape)

    raw_input = df2.to_numpy().flatten()
    print("Raw data", type(raw_input), raw_input.shape)

    bp = df3.to_numpy().flatten()
    print("Raw data", type(bp), bp.shape)

    samples = 15000
    time = np.linspace(0, 0.002, samples)
    y1 = raw_input[:samples]
    y2 = lp_filter_output[:samples]
    y3 = bp[:samples]

    # Plot the results
    plt.plot(time, y1, 'r--', time, y2, 'b--', time, y3, 'g--')
    plt.xlabel('time (s)')
    plt.ylabel('Unfiltered vs Data filtered with lp filter')
    plt.show()