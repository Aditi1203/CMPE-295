import wfdb as wf
import numpy as np
from glob import glob
import random
from matplotlib import pyplot as plt

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
        print(r2r_avg)

        patient = []

        # half of average of r2r_avg. it is used to locate starting and ending index on the left and right side of the r peak
        distance = r2r_avg/2
        for peak in r_peaks:
            i = int(peak - distance) if (peak - distance) > 0 else 0
            j = int(peak + distance) if (peak + distance) < len(distilled_record) else len(distilled_record)
            
