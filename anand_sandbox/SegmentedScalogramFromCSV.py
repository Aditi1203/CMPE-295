#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
        Script to generate Scalograms from CSV files.
            1. Change variable path to folder loacation of the dataset.
            2. Change variable path1 to path+/Scalogram
            3. Change number of items
        
        Tip: If program crashes, and suppose 2 subjects are completed, copy those 2 subject scalograms 
        and save it somewhere else. 
"""

import pandas as pd
import numpy as np
#import cv2
import os
#import imutils
from PIL import Image
from skimage import io
import imutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pywt
from scipy import signal
#import ecg_plot

def load_data(number_of_items=100):

    #-----------------------------------Change variable here------------------
    path = "pan_tomp_data"
    
    if not os.path.exists(path+"/Scalogram"):
        os.makedirs(path+"/Scalogram")
    
    #-----------------------------------Change variable here------------------
    path1= "pan_tomp_data/Scalogram"
    data = [] 
    curated_data = {"segments":[]}
    
    for subject_name in os.listdir(path)[:number_of_items]:
        if not os.path.exists(path1+"/"+subject_name):
            os.makedirs(path1+"/"+subject_name)
        
        if subject_name == ".DS_Store":
            continue
        if subject_name == "Scalogram":
            continue
            
        print ("Going through subject:" + subject_name)
        
        base=os.path.basename(path+"/"+subject_name)
        labelData=os.path.splitext(base)[0]
        
        print(labelData)
        i=0
        for items in os.listdir(path+"/"+subject_name):
            if items == ".DS_Store":
                continue
            
            else:
                try:
                    if items.endswith(".csv"):
                        i=i+1
                        print(str(i)+" begin")
                        scalName=str(items)+".png"
                        df1=np.genfromtxt(path+"/"+subject_name+"/"+items, delimiter=',')
                        #print(type(df1))
                        #-----------------------To show Segment plot. Comment scalogram plt---
                        #plt.plot(df1)
                        #plt.show()
                        #-----------------------------------------------
                        im1=df1
                        cwtmatr, freqs = pywt.cwt(im1, 14, 'mexh')
                        plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto', 
                        vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())  # doctest: +SKIP
                        #-------------To display scalogram------------
                        #plt.show() # doctest: +SKIP
                        #-----------------------------------------------
                        plt.savefig(path1+"/"+subject_name+"/"+scalName) 
                        #image1=cv2.imread(path+"/Scalogram"+"/"+subject_name+"/"+items)
                        #-------To display saved png image------------------------
                        #plt.imshow(image1)
                        #plt.show()
                        #------------------------------------------------------

                        curated_data['segments'].append(df1)
              #print(df1)
                except:
                    df = None  
        #data.append(data_model) # Save all the data

    return curated_data

#-----------------------------------Change variable here------------------

#--------------------Change number_of_items to 24 or as many number of people you want to convert---
data = load_data(number_of_items=2)

