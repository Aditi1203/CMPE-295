import os
import wfdb as wf
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

RAW_SIGNAL_PATH = "RAW_ECG_CSV"
FILTERED_DATA_PATH = "BANDPASS_LP5_HP_50"


def get_paths():
    """
    Generate Path location of files
    """
    paths = glob("CYBHi/data/long-term/*")  # list of all the data files.
    return paths


def generate_raw_csv(filepaths, RAW_SIGNAL_PATH):
    """
    Read the files and and save the raw ecg values in CSV file
    :param filepaths: Path of the files
    :param RAW_SIGNAL_PATH: Base Folder where to save the files
    :return:
    """
    a = set()
    for path in filepaths:

        # 20120111-MP-A0-35.txt Take MP for example
        file_folder_name = path.rsplit("-")[2]
        print(file_folder_name)
        if file_folder_name in a:
            continue
        else:
            a.add(file_folder_name)

        file_path = os.path.join(RAW_SIGNAL_PATH, file_folder_name + ".csv")

        data_value = np.loadtxt(path)
        data_value = data_value[:120000]

        np.savetxt(file_path, data_value, delimiter=",", fmt="%.2f")


if __name__ == "__main__":
    # Get Path of the files
    file_names = get_paths()

    if os.path.exists(RAW_SIGNAL_PATH):
        print("The directory exists")
    else:
        print("Creating directory")
        os.makedirs(RAW_SIGNAL_PATH)

    generate_raw_csv(file_names, RAW_SIGNAL_PATH)

    if os.path.exists(FILTERED_DATA_PATH):
        print('The directory exists')
    else:
        print("Creating directory")
        os.makedirs(FILTERED_DATA_PATH)

    from butterworth import read_signals
    read_signals(file_names, FILTERED_DATA_PATH)
