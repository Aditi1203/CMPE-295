import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import shutil

def one_hot_encoding(qrs_complex, record_name):
    '''
    A one hot encoding is a representation of categorical variables as binary vectors.

    First step requires that the categorical values be mapped to integer values.

    Then, each integer value is represented as a binary vector that is all zero values except the index of the integer,
    which is marked with a 1.

    '''
    print("file-------file", record_name)
    file_path = os.getcwd() + '/encoded_segments/'

    if os.path.exists(file_path):
        print("encoded_segments folder exists")
    else:
        print("Creating folder: encoded_segments")
        os.makedirs(file_path)

    sub_path = os.path.join(file_path,record_name)
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)

    if os.path.exists(sub_path + '.zip'):
        print('{0}\'s directory already exists'.format(record_name))
        return

    count=0
    sequence_path=sub_path+"/sequence"
    for qrs in qrs_complex:
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(qrs)
        # print("Integer Encoded", integer_encoded)
        # Binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        # print(onehot_encoded)
        np.save(sequence_path + str(count), onehot_encoded)  # Saving as a binary file (numpy array)
        count = count + 1

    # Remove the folder and save only zip file
    shutil.make_archive(sub_path, 'zip', sub_path)
    shutil.rmtree(sub_path)