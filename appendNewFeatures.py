# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 08:02:48 2019

Extract new features from existing list of images and append to old list.
Plan to use to make predictions about popularity of these images based on these features.

@author: Anthony Schramm
"""

import cv2
import numpy as np
import os
import pandas as pd
 
def getList(dir):
    idList = []
    for filename in os.listdir(dir):
        basename, file_extension = os.path.splitext(filename)
        idList.append([basename,file_extension])
    return idList

imgFolder = 'ImagesCompile'
idList = getList(imgFolder)

old_data = pd.read_csv('imageFeatures_All_new.csv',encoding = "ISO-8859-1")

#open dictionary, access keys in order, create a new dictionary
#add new keys to new dictionary, append to new dictionary in order
#after for loop merge dictionaries, should be empty but new values will go in order should be correct

topics_dict = { "coarseContrast":[], \
                "bluHue": [], \
                "redHue": [], \
                "grnHue": [], \
                "id": []}
i = 0
print(len(idList))
for img in idList:
    i += 1
    imgID = img[0]
    imgExt = img[1]
              
        
    imgDir = imgFolder + '/' + imgID + imgExt
    # read image into matrix.
    img =  cv2.imread(imgDir)
    img = img / 255 #0 to 1 scale
    img = img + 0.0001 #pad to avoid conditional for saturation on pure black pixels
    
    
    
    #Brightness and brightness balance
    luminance = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2] #Relative luminance (https://en.wikipedia.org/wiki/Relative_luminance)
    redHue = np.mean(0.2126*img[:,:,0])
    grnHue = np.mean(0.7152*img[:,:,1])
    bluHue = np.mean(0.0722*img[:,:,2])
    coarseContrast = np.var(luminance)
    #Left/right and top/bottom luminance balance
    
    
    topics_dict["redHue"].append(redHue)
    topics_dict["grnHue"].append(grnHue)
    topics_dict["bluHue"].append(bluHue)
    topics_dict["coarseContrast"].append(coarseContrast)
    topics_dict["id"].append(imgID)
    print(i)
    img = None
    #luminance symmetry line?
old_data = pd.read_csv('imageFeatures_All_new.csv',encoding = "ISO-8859-1")

topics_data = pd.DataFrame(topics_dict)

new_data = pd.merge(old_data, topics_data, on='id')
#Remove duplicate rows, if any
#new_data = new_data.drop_duplicates(subset = 'id')

new_data.to_csv('imageFeatures_All_newfeat.csv', index=False)
