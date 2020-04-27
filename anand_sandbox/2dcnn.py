# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import pandas as pd
import numpy as np
import cv2
import os
import imutils
from keras.applications import VGG19, VGG16
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import keras
from keras import Sequential, losses, optimizers, Input
from keras.layers import Dense, Conv2D, MaxPool2D, Average, Flatten, Dropout, Activation
from keras.utils import to_categorical, plot_model,vis_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard, EarlyStopping
from keras.preprocessing.image import img_to_array
from skimage import io
import imutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pywt
from scipy import signal


# %%
class DataModel:
    """
            DataModel
            not used currently in the program.
    """
    def __init__(self):
        self.items_labels = None
        self.images = []
        self.curated_data = {}
    def add_image(self, image_array):
        self.images.append(image_array)
    def add_labels(self, df):
        self.item_labels = df
    def add_curated_data(self, image, emotion):
        self.curated_data['segment'] = image #Array from csv of each segment
        self.curated_data['label'] = label #Label
        self.curated_data['scalogram'] =scaledImage #scalogram of each segment

def load_data(number_of_items=90):
    """
        number_of_items -> Number of items to return
        returns the data in a dictionary of images and labels.
    """
    path = "scalogram_module/Scalogram"
    path1= "scalogram_module/Scalogram/Cropped"
    data = [] 
    curated_data = {"label":[], "scalogram":[]}
    for subject_name in os.listdir(path)[:number_of_items]:
        # At the start of the iteration build a data model
        data_model = DataModel()
        if subject_name == ".DS_Store":
            continue
        if subject_name  ==".ipynb_checkpoints":
            continue
        if subject_name  ==".svn":
            continue
        print ("Going through subject:" + subject_name)
        base=os.path.basename(path+"/"+subject_name)
        labelData=os.path.splitext(base)[0]
        print(labelData)
        i=0
        for items in os.listdir(path+"/"+subject_name):
            if items == ".DS_Store":
                continue
            if items.endswith(".png"):
                #i=i+1
                #print(i)
                try:
                    im2 = cv2.imread(path+"/"+subject_name+"/"+items)
                    #plt.imshow(im2)
                    #plt.show()
                    crop_img = im2[30:20+235, 50:50+342]
                    im = cv2.resize(crop_img, (224,224)) # Changing into 80x80X3
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    #print(type(crop_img))
                    #plt.imshow(crop_img)
                    #plt.show()
                    

              #curated_data['segments'].append(df1)
                    #curated_data['scalogram'].append(crop_img)
                    curated_data['scalogram'].append(im)
                    curated_data['label'].append(labelData)
              #print(df1)
                except:
                      df = None  
        #data.append(data_model) # Save all the data

    return curated_data


# %%
data = load_data(number_of_items=48)


# %%
temp=np.array(data['scalogram'])
print(temp.shape)


# %%
encoder = LabelEncoder()
encoder.fit(data['label'])


# %%
data['label'] = encoder.transform(data['label'])


# %%
all_the_classes = encoder.classes_
mapping = {0: 'person_100',
 1: 'person_101',
 2: 'person_102',
 3: 'person_103',
 4: 'person_104',
 5: 'person_105',
 6: 'person_106',
 7: 'person_107',
 8: 'person_108',
 9: 'person_109',
 10: 'person_111',
 11: 'person_112',
 12: 'person_113',
 13: 'person_114',
 14: 'person_115',
 15: 'person_116',
 16: 'person_117',
 17: 'person_118',
 18: 'person_119',
 19: 'person_121',
 20: 'person_122',
 21: 'person_123',
 22: 'person_124',
 23: 'person_200',
 24: 'person_201',
 25: 'person_202',
 26: 'person_203',
 27: 'person_205',
 28: 'person_207',
 29: 'person_208',
 30: 'person_209',
 31: 'person_210',
 32: 'person_212',
 33: 'person_213',
 34: 'person_214',
 35: 'person_215',
 36: 'person_217',
 37: 'person_219',
 38: 'person_220',
 39: 'person_221',
 40: 'person_222',
 41: 'person_223',
 42: 'person_228',
 43: 'person_230',
 44: 'person_231',
 45: 'person_232',
 46: 'person_233',
 47: 'person_234'    
          }

#print(mapping)


# Get emotion from class number   
def get_name_from_class(class_number):
    """
        gets the corresponding subject from the class
    """
    if mapping.get(class_number,None):
        return mapping.get(class_number)
    else:
        return -1 # No such class


# %%
# Extracting features and labels
features = np.array(data['scalogram'])
labels = data['label']


# %%
print ("Shape of features: ", features.shape)
print ("Shape of labels: ", labels.shape)


# %%
# Changing
labels = to_categorical(labels, num_classes=len(mapping))


# %%
# Building model
model = Sequential()
model.add(Conv2D(32, kernel_size=5, input_shape=(224, 224, 1), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, kernel_size=5, activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
#model.add(Dense(150, activation="relu"))
#model.add(Dense(120, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(80, activation="relu"))
#model.add(Dropout(0.5))
model.add(Dense(len(mapping), activation="softmax"))


# %%
model.summary()


# %%
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizers.RMSprop(),metrics=['accuracy'])


# %%
# Splitting
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, shuffle=True, random_state=17)


# %%
tensorboard = TensorBoard()
earlystopping = EarlyStopping(patience=3)


# %%
# Scaling
features_train = features_train.astype("float32")
features_test = features_test.astype("float32")


# %%
features_train = features_train / 1/255
features_test = features_test / 1/255


# %%
features_train = features_train.reshape(len(features_train), 224, 224, 1)
features_test = features_test.reshape(len(features_test), 224, 224, 1)


# %%
# Training
model.fit(features_train, labels_train, epochs=100, batch_size=16, callbacks=[tensorboard,earlystopping], validation_data=(features_test, labels_test))


# %%
model.save("2DCNNButterworthExtended.h5") # Saving the model


# %%
# Test an image
def test_image(image,label):
    #print(image.shape)
    prediction = np.argmax(model.predict(image.reshape(1,224,224,1)))
    result_predict=get_emotion_from_class(prediction)
    actual=np.argmax(label)
    result_actual=get_emotion_from_class(actual)
    #print(prediction)
    #print(actual)
    return result_actual,result_predict

    
# Get emotion from class number   
def get_emotion_from_class(class_number):
    """
        gets the corresponding label from the class
    """
    if mapping.get(class_number,None):
      return mapping.get(class_number)
    else:
      return -1 # No such class


# %%
for i in range(len(features_test)):
    actual,predicted=test_image(features_test[i],labels_test[i])
    print("Model predicts: "+predicted+" and actual label is: "+actual)
    if(predicted!=actual):
        print("Wrong prediction")
    print("-------------------------------------------------------------")

