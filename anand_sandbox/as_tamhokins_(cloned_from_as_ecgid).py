import wfdb as wf
import numpy as np
from glob import glob
import random
from matplotlib import pyplot as plt
import os

from helper import save_to_csv

# experimental package: adding a package for detecting r peaks using pan tomkins algorithm
from ecgdetectors import Detectors
detectors = Detectors(360)

def get_paths():
    paths = glob("ecg-id-database/Person_**/*.atr")
    paths = [path[:-4] for path in paths]
    # print('Only paths', paths)
    return paths

def read_signals(paths):
    all_signals = []
    for rec in paths:
        record_number = rec[23:25]
        segment_folder = "pan_tomp_data/person_"+str(record_number)
        if os.path.exists(segment_folder):
            print('The directory exists')
        else:
            print("Creating directory")
            os.makedirs(segment_folder)
        # os.makedirs("pan_tomp_data/person_"+str(record_number))
        record = wf.rdsamp(rec, channels=[0])
        p_signals, fields = wf.rdsamp(rec, channels=[0])
        num_cols = len(record)
        num_rows = len(record[0])
        annotation = wf.rdann(rec, "atr")
        all_beats = annotation.sample[:]
        beats = annotation.sample

        # detecting qrs using pan tompkins via py-ecg-detector package
        distilled_record = [a[0] for a in record[0]]
        r_peaks = detectors.pan_tompkins_detector(distilled_record)

        r2r_sum = 0
        for i in range(len(r_peaks)-1):
            r2r = r_peaks[i+1]-r_peaks[i]
            r2r_sum += r2r
        
        r2r_avg =r2r_sum/len(r_peaks)

        patient = []

        # half of average of r2r_avg. it is used to locate starting and ending index on the left and right side of the r peak
        distance = r2r_avg/2
        count = 0
        for peak in r_peaks:
            count += 1
            i = int(peak - distance) if (peak - distance) > 0 else 0
            j = int(peak + distance) if (peak + distance) < len(distilled_record) else len(distilled_record)
            segment = distilled_record[i:j+1]
            segment_for_csv = np.array(segment, dtype=np.float)
            # print("length of segment:",len(segment))
            save_to_csv(segment_for_csv,"pan_tomp_data/person_"+str(record_number)+"/seg"+str(count)+".csv")


paths = get_paths()
try:
    read_signals(paths)
    from helper import save_to_csv, create_signal, signal_to_image, input_files, create_data_label
    # save_to_csv(segment, "ecgid_data_segmented.csv")
except FileNotFoundError:
    print(".hea file not found")
