# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 08:02:48 2019

Extract features from new list of images. Add to old feature list. 
Plan to use to make predictions about popularity of these images based on these features.

@author: Anthony Schramm
"""

import cv2
import numpy as np
import os
import praw
import pandas as pd
import datetime as dt
 
def getList(dir):
    idList = []
    for filename in os.listdir(dir):
        basename, file_extension = os.path.splitext(filename)
        idList.append([basename,file_extension])
    return idList

imgFolder = 'DSImages_used190321'
idList = getList(imgFolder)
old_data = pd.read_csv('imageFeatures_190314.csv',encoding = "ISO-8859-1")

reddit = praw.Reddit(client_id='yArkbXtfNCeTOw', \
                     client_secret='j2jbgoLCOu32qXO6vaE1oZ53JtA', \
                     user_agent='Landscape Scraper', \
                     username='schraper', \
                     password='DSreddIt')

#open dictionary, access keys in order, create a new dictionary
#add new keys to new dictionary, append to new dictionary in order
#after for loop merge dictionaries, should be empty but new values will go in order should be correct

topics_dict = { "title":[], \
                "score":[], \
                "id":[], "url":[], \
                "comms_num": [], \
                "created": [], \
                "aspRatio":[], \
                "imgSz":[], "saturation":[], \
                "luminance": [], \
                "lumLRBal": [], \
                "lumTBBal": [], \
                "globalContrast": [], \
                "redLRBal": [], \
                "redTBBal": [], \
                "grnLRBal": [], \
                "grnTBBal": [], \
                "bluLRBal": [], \
                "bluTBBal": [], \
                "redHue": [], \
                "grnHue": [], \
                "bluHue": [], \
                "coarseContrast": []}
i = 0
print(len(idList))
for img in idList:
    i += 1
    imgID = img[0]
    imgExt = img[1]
    if not(imgID in old_data.id.values):
        submission = reddit.submission(id=imgID)
        topics_dict["id"].append(submission.id)
        topics_dict["title"].append(submission.title)
        topics_dict["score"].append(submission.score)
        topics_dict["url"].append(submission.url)
        topics_dict["comms_num"].append(submission.num_comments)
        topics_dict["created"].append(submission.created)
        
        
        imgDir = imgFolder + '/' + imgID + imgExt
        # read image into matrix.
        img =  cv2.imread(imgDir)
        img = img / 255 #0 to 1 scale
        img = img + 0.0000001 #pad to avoid conditional for saturation on pure black pixels
        
        # get image properties.
        h,w,bpp = np.shape(img)
         
    #    # print image properties.
    #    print("width: " + str(w))
    #    print("height: " + str(h))
    #    print("bpp: " + str(bpp))
        
        # Want to extract:
        #Extract color values
        #rPrime = img[:,:,0] / 255
        #gPrime = img[:,:,1] / 255
        #bPrime = img[:,:,2] / 255
        
        cMax = np.zeros((h,w))
        cMaxInd = np.zeros((h,w))
        
        maxVals = np.amax(img, axis = 2)
        cMaxInd = np.argmax(img, axis = 2)
        minVals = np.amin(img, axis = 2)
        cMinInd = np.argmin(img, axis = 2)
        deltaCol = maxVals - minVals
        
        #Aspect ratio
        aspRat = w/h
        
        #Image size (MP)
        imSz = w*h
        hlfW = w//2
        hlfH = h//2
        
        # Saturation
        saturation = deltaCol / maxVals
        avgSat = np.mean(saturation)
        
        #Brightness and brightness balance
        luminance = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2] #Relative luminance (https://en.wikipedia.org/wiki/Relative_luminance)
        avgLum = np.mean(luminance)
        redHue = np.mean(0.2126*img[:,:,0])
        grnHue = np.mean(0.7152*img[:,:,1])
        bluHue = np.mean(0.0722*img[:,:,2])
        overallContrast = np.var(luminance)
        #Left/right and top/bottom luminance balance
        lumLRBal = np.sum(luminance[:, :hlfW]) / np.sum(luminance[:, hlfW:])
        lumTBBal = np.sum(luminance[:hlfH, :]) / np.sum(luminance[hlfH:, :])
        
        #Contrast - assume for now no gamma correction needed
        
        #Local contrast/global contrast function https://pdfs.semanticscholar.org/c7ee/519d5e7378e64fc08c06c3d81250c99867f3.pdf
        #Avoid edges to simplify calculation
        localContrast = (np.absolute(luminance[1:-1, 1:-1] - luminance[0:-2, 1:-1]) + 
                         np.absolute(luminance[1:-1, 1:-1] - luminance[2:, 1:-1]) + 
                         np.absolute(luminance[1:-1, 1:-1] - luminance[1:-1, :-2]) + 
                         np.absolute(luminance[1:-1, 1:-1] - luminance[1:-1, 2:])) / 4
        globalContrast = np.mean(localContrast)
        
        #Color balance (LR, TB)
        redLRBal = np.sum(img[:, :hlfW, 0]) / np.sum(img[:, hlfW:, 0])
        redTBBal = np.sum(img[:hlfH, :, 0]) / np.sum(img[hlfH:, :, 0])
        grnLRBal = np.sum(img[:, :hlfW, 1]) / np.sum(img[:, hlfW:, 1])
        grnTBBal = np.sum(img[:hlfH, :, 1]) / np.sum(img[hlfH:, :, 2])
        bluLRBal = np.sum(img[:, :hlfW, 2]) / np.sum(img[:, hlfW:, 2])
        bluTBBal = np.sum(img[:hlfH, :, 2]) / np.sum(img[hlfH:, :, 2])
        
        
        topics_dict["aspRatio"].append(aspRat)
        topics_dict["imgSz"].append(imSz)
        topics_dict["saturation"].append(avgSat)
        topics_dict["luminance"].append(avgLum)
        topics_dict["lumLRBal"].append(lumLRBal)
        topics_dict["lumTBBal"].append(lumTBBal)
        topics_dict["globalContrast"].append(globalContrast)
        topics_dict["redLRBal"].append(redLRBal)
        topics_dict["redTBBal"].append(redTBBal)
        topics_dict["grnLRBal"].append(grnLRBal)
        topics_dict["grnTBBal"].append(grnTBBal)
        topics_dict["bluLRBal"].append(bluLRBal)
        topics_dict["bluTBBal"].append(bluTBBal)
        topics_dict["redHue"].append(redHue)
        topics_dict["grnHue"].append(grnHue)
        topics_dict["bluHue"].append(bluHue)
        topics_dict["coarseContrast"].append(overallContrast)
        print(i)
        #luminance symmetry line?
    
topics_data = pd.DataFrame(topics_dict)

def get_date(created):
    return dt.datetime.fromtimestamp(created)

_timestamp = topics_data["created"].apply(get_date)

topics_data = topics_data.assign(timestamp = _timestamp)

new_data = pd.concat([old_data, topics_data])
#Remove duplicate rows, if any
new_data = new_data.drop_duplicates(subset = 'id')

new_data.to_csv('imageFeatures_190321.csv', index=False)
