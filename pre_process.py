from IPython.display import display
import pandas as pd
import numpy as np
import wfdb
import matplotlib.pyplot as plt
import os, sys

class conversion:
    def __init__(self):
        self.dir = os.path.join(os.getcwd(), 'data1')
        self.database = 'ecgiddb'
        # record = wfdb.rdrecord('ecg-id-database-1.0.0/Person_01/rec_1')
        # wfdb.plot_wfdb(record=record, title='Record a103l from Physionet Challenge 2015')
        # display(record.__dict__)

    def constructor(self, folder, filename):
        signals, fields = wfdb.rdsamp(filename, sampfrom=0, pb_dir=os.path.join(self.database, folder))
        df = pd.DataFrame(signals)
        df.to_csv(os.path.join(self.dir, folder, filename + "." 'csv'), index=False)

    def tocsv(self):
        for folders in os.listdir(self.dir):
            if (folders.startswith('Person_')):
                for inpersonsdir in os.listdir(os.path.join(self.dir, folders)):
                    if (inpersonsdir.endswith('dat')):
                        basename = inpersonsdir.split(".", 1)[0]
                        self.constructor(folders, basename)


class ProcessData:
    def __init__(self):
        self.dir = os.path.join(os.getcwd(), 'data1')
        self.persons_labels = []  # who the person is
        self.age_labels = []  # age of thatperson
        self.gender_labels = []  # is that person male or female
        self.date_labels = []  # month.day.year of ecg record
        self.ecg_filsignal = pd.DataFrame()  # filtered ecg dataset
        self.ecg_signal = pd.DataFrame()  # unfiltered ecg dataset

    # extracts labels and features from rec_1.hea of each person
    def extract_labels(self, filepath):
        for folders in os.listdir(filepath):
            if (folders.startswith('Person_')):
                self.persons_labels.append(folders)
                for inpersonsdir in os.listdir(os.path.join(filepath, folders)):
                    if (inpersonsdir.startswith('rec_1.') and inpersonsdir.endswith('hea')):
                        with open(os.path.join(filepath, folders, inpersonsdir), "r") as f:
                            array2d = [[str(token) for token in line.split()] for line in f]
                            self.age_labels.append(array2d[4][2])
                            self.gender_labels.append(array2d[5][2])
                            self.date_labels.append(array2d[6][3])
                        f.close()

    def extract_feats(self, filepath):
        p = 0  # person counter
        global f_num
        f_num = 0  # file counter
        for folders in os.listdir(filepath):
            if (folders.startswith('Person_')):
                p = p + 1
                for files in os.listdir(os.path.join(filepath, folders)):
                    if (files.endswith('csv')):
                        with open(os.path.join(filepath, folders, files), "r") as x:
                            f_num = f_num + 1
                            features = pd.read_csv(x, header=[0, 1])
                            pdfeats = pd.DataFrame(features)
                            pdfeats = pdfeats.apply(pd.to_numeric)
                            temp = [p]  # 0th index is person_label int
                            temp1 = [p]
                            for rows in range(len(pdfeats)):
                                temp.append(pdfeats.get_value(rows, 1, True))
                                temp1.append(pdfeats.get_value(rows, 0, True))
                            tempnp = np.asarray(temp, dtype=float)
                            if (tempnp.shape == (9999,)):
                                tempnp = np.append(tempnp, tempnp[9998])
                            temp1np = np.asarray(temp1, dtype=float)
                            if (temp1np.shape == (9999,)):
                                temp1np = np.append(temp1np, tempnp[9998])
                            self.dumpfeats(tempnp, 1)
                            self.dumpfeats(temp1np, 2)
                        x.close()

        # appends to a bigger global array

    def dumpfeats(self, array, flag):
        fil_df = pd.DataFrame(array)
        fil_df = fil_df.T
        ufil_df = pd.DataFrame(array)
        ufil_df = ufil_df.T
        if (flag == 1):
            self.ecg_filsignal = self.ecg_filsignal.append(fil_df, ignore_index=True)
        if (flag == 2):
            self.ecg_signal = self.ecg_signal.append(ufil_df, ignore_index=True)

    def init(self):
        print("Setting up DeepECG data1 labels..")
        self.extract_labels(self.dir)
        ecglabels = [list(i) for i in zip(self.persons_labels, self.age_labels, self.gender_labels, self.date_labels)]
        print("Exporting labels to csv..")
        df_ecglabels = pd.DataFrame(ecglabels)
        df_ecglabels.to_csv(os.path.join('processed_data', 'ecgdblabels.csv'), index=False)
        print("Export complete.")

        print("Setting up DeepECG data1 features..")
        self.extract_feats(self.dir)
        print("Exporting feature set to csv..")
        self.ecg_filsignal.to_csv(os.path.join('processed_data', 'filecgdata.csv'), index=False)
        self.ecg_signal.to_csv(os.path.join('processed_data', 'unfilecgdata.csv'), index=False)
        print("Export complete.")

        if (os.path.isfile(os.path.join('processed_data', 'filecgdata' + "." + 'csv')) and os.path.isfile(
                os.path.join('processed_data', 'unfilecgdata' + "." + 'csv'))):
            print("Data in processed_data/ folder is now ready for training.")

def main():
    generate=conversion()
    generate.tocsv()

    processing = ProcessData()
    processing.init()

if __name__=='__main__':
    main()