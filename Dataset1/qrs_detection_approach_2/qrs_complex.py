import numpy as np
import pandas as pd
import biosppy
import os
import matplotlib.pyplot as plt
import cv2
from glob import glob
import random
import tensorflow as tf
import errno

segment_csv_main="segmented_csv"
def save_to_csv(signals, file_list, file):
    # Save to CSV file.
    # print(list(beats[:]))
    # savedata = np.array(list(beats[:]), dtype=np.float)
    # for f in file_list:
        # print("File name-----------", f)
        #
        # directory="segmented_csv"+f;
        # if os.path.exists(directory):
        #     print('The directory exists')
        # else:
        #     print("Creating directory")
        #     os.makedirs(directory)
        #
        # file_name = file
        #
        # print("---------------------------",len(signals), signals)
        # filename = directory + '/' + file_name
        # with open(file_name, "wb") as fi:
        #     np.savetxt(fi, signals, delimiter=",", fmt='%f')
        #     print(' File created: ', filename)

    file_name = file
    for f in file_list:

        directory = segment_csv_main + f;
        filename = directory + '/' + file_name
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(filename, "w") as f:
            np.savetxt(f, signals, delimiter=",", fmt='%f')
            print(' File created: ', filename)


def create_signal(file_list, file_name, signal, segment):
    for f in file_list:
        file=segment_csv_main+f+ '/' + file_name
        print("file name in create_signal",file)
        csv = pd.read_csv(file, names=["values"])
        print("csv", csv)
        data = np.array(csv['values'])
        print("data1", data)
        print("signal", signal)
        print("segment", segment)
        print("shape", data.shape)
        print("data1 len", len(data))
        signals = []
        count = 1
        peaks = biosppy.signals.ecg.christov_segmenter(signal=segment, sampling_rate=200)[0]
        for i in (peaks[1:-1]):
            diff1 = abs(peaks[count - 1] - i)
            diff2 = abs(peaks[count + 1] - i)
            x = peaks[count - 1] + diff1 // 2
            y = peaks[count + 1] - diff2 // 2
            signal = data[x:y]
            signals.append(signal)
            count += 1
        signal_to_image(signals,"Images",f)
    return signals

def signal_to_image(array, directory,record_number):

    basic_folder = "Dataset"
    record_no = record_number.rsplit("/", -1)[1]
    secondary_folder = basic_folder + "/person" + str(record_no)
    # +"Person"+str(record_number)+"/"+str(count)
    if os.path.exists(secondary_folder):
        print('The directory exists')
    else:
        print("Creating directory")
        os.makedirs(secondary_folder)

    text_file = secondary_folder + '/person' + str(record_no) + '.txt'
    file = open(text_file, "w")
    file.write("Person 1")
    file.close()
    # with open(text_file, "w") as f:
    #     np.savetxt("Person", record_no)

    for count, i in enumerate(array):
        fig = plt.figure(frameon=False)
        plt.plot(i)
        plt.xticks([]), plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        filename = secondary_folder + '/segment' + str(count+1) + '.png'
        fig.savefig(filename)
        im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        im_gray = cv2.resize(im_gray, (128, 128), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(filename, im_gray)
        plt.close()


    # return directory

def input_files(mainData):
    print("here")
    nData = glob(mainData + '/*.png')

    n_formatted = []

    for n in nData:
        n_formatted.append(n + " 1")

    print(n_formatted)
    random.shuffle(n_formatted)

    first_half_n_formatted = n_formatted[:1000]
    print(first_half_n_formatted)

    testFiles = first_half_n_formatted
    trainFiles = n_formatted[1000:]

    with open('training_files.txt', 'w') as train:
        for t in trainFiles:
            train.write(t + '\n')

    with open('testing_files.txt', 'w') as test:
        for t in testFiles:
            test.write(t + '\n')


def create_data_label(filename):
    file = open(filename, 'r')
    lines = file.readlines()

    x_train = np.zeros((len(lines), 4, 128, 128, 1))
    x_label = np.zeros(len(lines), dtype='int')

    for i in range(len(lines)):
        print("loading image: " + str(i))
        path = lines[i].split(" ")[0]
        label = lines[i].split(" ")[-1]

        label = label.strip('\n')
        label = int(label)
        x_label[i] = label

        img_raw = tf.io.read_file(path)
        img_tensor = tf.image.decode_image(img_raw)
        img_final = tf.image.resize(img_tensor, [128, 128])
        # print(img_final)
        # print([128] + img_tensor)
        # print(img_tensor.shape)
        x_train[i] = [128] + img_final
        # print(x_train[0].dtype)
        # print(x_train[1].dtype)
        # print(x_train[2].dtype)
        # print(x_train[3].dtype)
        # print(x_train[4].dtype)
    # print(len(x_train))
    return x_train, x_label
