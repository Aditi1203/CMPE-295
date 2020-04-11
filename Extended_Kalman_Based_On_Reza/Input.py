import os
import wfdb as wf
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

RAW_SIGNAL_PATH='RAW_ECG_CSV'

def get_paths():
    paths = glob('data1/*.atr')  # list of all the data files.

    # Get rid of the extension and sort in increasing order.
    paths = [path[:-4] for path in paths]
    paths.sort()
    return paths

def read_signals(paths):

    # record=0
    for rec in paths:
         record_no = rec.rsplit("/", -1)[1]
         print("here", record_no)
         # record = wf.rdsamp(rec, channels=[0])
    return record_no
#     p_signals, fields = wf.rdsamp(rec, channels=[0])


if __name__ == "__main__":

    paths = get_paths()
    print("Path of the returned ECG recording/s:")
    # print(paths)
    signals = read_signals(paths)
    # print(signals, type(signals))
    # num_cols = len(signals)
    # num_rows = len(signals[0])
    # print("Shape of rows and column ofs record:", num_rows, num_cols)
    # signals=np.asarray(signals[0]).flatten()
    # print(type(signals), signals.shape)
    #

    # np.savetxt('test_input.csv', signals, fmt='%2.4e')




    # import pandas as pd
    #
    # path1='Filter_0.5/EKF5.csv'
    # path2 = 'Filter_0.5/EKS5.csv'
    # df1 = pd.read_csv(path1, sep=' ', header=None)
    # df2 = pd.read_csv(path2, sep=' ', header=None)
    # print("Read from csv", df1.values)
    # ekf= df1.to_numpy().flatten()
    # eks = df2.to_numpy().flatten()
    # print("ekf style", type(ekf), ekf.shape)
    #
    # # x1 = np.linspace(0.0, 5000.0)
    # # x2 = np.linspace(0.0, 5000.0)
    # #
    # # y1 = signals[:5000]
    # # y2 = signals[:5000]
    # #
    # # plt.subplot(2, 1, 1)
    # # plt.plot(x1, y1, 'o-')
    # # plt.title('Unfiltered vs Extended Kalman Filter vs EF Smoothening')
    # # plt.ylabel('Unfiltered')
    # #
    # # plt.subplot(2, 1, 2)
    # # plt.plot(x2, y2, '.-')
    # # plt.xlabel('time (s)')
    # # plt.ylabel('EKF')
    # #
    # # plt.show()
    #
    # # Solution 1:
    # narr = ekf[:3000]
    # time = np.linspace(0, 0.02, 3000)
    # plt.plot(time, narr)
    # plt.show()
    #
    # # Solution 2:
    # time = np.linspace(0, 0.2, 3000)
    # y1 = signals[:3000]
    # y2 = ekf[:3000]
    # y3 = eks[:3000]
    #
    # plt.subplot(2, 1, 1)
    # plt.plot(time, y1, 'o-')
    # plt.title('Unfiltered vs Extended Kalman Filter vs EF Smoothening')
    # plt.ylabel('Unfiltered')
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(time, y2, '.-')
    # plt.xlabel('time (s)')
    # plt.ylabel('EKF')
    #
    # plt.show()
    #
    # # Solution 3:
    # # plt.scatter(time, y1, color='k--')
    # # plt.scatter(time, y2, color='g--')
    # # plt.show()
    #
    # t = np.arange(0., 5., 0.2)
    #
    # # red dashes, blue squares and green triangles
    # plt.plot(time, y1, 'r--', time, y2, 'b--', time, y3 , 'y--')
    # plt.xlabel('time (s)')
    # plt.ylabel('Unfiltered vs Extended Kalman Filter vs EF Smoothening with Bandpass filter')
    # plt.show()
    #
    # # 2nd graph
    # path1 = 'No_Filter/EKF_no.csv'
    # path2 = 'No_Filter/EKS_no.csv'
    # df1 = pd.read_csv(path1, sep=' ', header=None)
    # df2 = pd.read_csv(path2, sep=' ', header=None)
    # print("Read from csv", df1.values)
    # ekf = df1.to_numpy().flatten()
    # eks = df2.to_numpy().flatten()
    #
    # y1 = signals[:3000]
    # y2 = ekf[:3000]
    # y3 = eks[:3000]
    #
    # plt.plot(time, y1, 'r--', time, y2, 'b--', time, y3, 'y--')
    # plt.xlabel('time (s)')
    # plt.ylabel('Unfiltered vs Extended Kalman Filter vs EF Smoothening without Bandpass filter')
    # plt.show()
