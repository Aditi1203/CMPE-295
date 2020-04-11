import numpy as np
import pandas as pd
import biosppy
import os
import matplotlib.pyplot as plt
import cv2
from glob import glob
import random
import tensorflow as tf

def save_to_csv(signals, file_name):
    with open(file_name, "w") as fi:
        np.savetxt(fi, signals, fmt="%f")
        print("file created: ",file_name)
    

def create_signal(file_name, signal, segment):
    csv = pd.read_csv(file_name, names=["values"])
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
    return signals

def signal_to_image(array, directory):
    if os.path.exists(directory):
        print('The directory exists')
    else:
        print("Creating directory")
        os.makedirs(directory)

    for count, i in enumerate(array):
        fig = plt.figure(frameon=False)
        plt.plot(i)
        plt.xticks([]), plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        filename = directory + '/' + str(count) + '.png'
        fig.savefig(filename)
        im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        im_gray = cv2.resize(im_gray, (128, 128), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(filename, im_gray)
        plt.close()
    return directory

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