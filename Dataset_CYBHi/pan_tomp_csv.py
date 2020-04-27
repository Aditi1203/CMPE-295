import wfdb as wf
import numpy as np
from glob import glob
import random
from matplotlib import pyplot as plt
import os
import csv

from helper import save_to_csv

# experimental package: adding a package for detecting r peaks using pan tomkins algorithm
from ecgdetectors import Detectors
detectors = Detectors(1000)

def get_paths():
    # paths = glob("ecg-id-database/Person_**/*.atr")
    paths = glob("BANDPASS_LP5_HP_50/*.csv")
    # paths = [path[:-4] for path in paths]
    paths.sort()
    return paths

def read_signals(paths):
    all_signals = []
    for rec in paths:
        
        record_number = (rec.split("/")[1]).split('.')[0]
        print(record_number)
        segment_folder = "butterworth_segments/person_"+str(record_number)
        if os.path.exists(segment_folder):
            print('The directory exists')
        else:
            print("Creating directory")
            os.makedirs(segment_folder)

        distilled_record = []
        with open(rec) as csvfile:
            record_reader = csv.reader(csvfile, delimiter = ',')
            for row in record_reader:
                distilled_record.append(float(row[0]))
                # distilled_record = [float(num) for num in row]
        
        
        # detecting qrs using pan tompkins via py-ecg-detector package
        print(distilled_record)
        r_peaks = detectors.pan_tompkins_detector(distilled_record)
        print("R_Peaks length:", len(r_peaks))

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
            path=os.path.join("butterworth_segments", "person_"+record_number+"/seg"+str(count)+".csv")
            save_to_csv(segment_for_csv,path)


paths = get_paths()
# try:
read_signals(paths)
from helper import save_to_csv, create_signal, signal_to_image, input_files, create_data_label
    # save_to_csv(segment, "ecgid_data_segmented.csv")
# except FileNotFoundError:
#     print(".hea file not found")