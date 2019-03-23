# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:57:44 2019

@author: night
"""
# Import function to create training and test set splits
from sklearn.cross_validation import train_test_split

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import time

def getTimestamp(date):
    return time.mktime(dt.datetime.strptime(date,"%d/%m/%Y").timetuple())

#Import data
dataFilename = 'imageFeatures_190321.csv'
imageDF = pd.read_csv(dataFilename,encoding = "ISO-8859-1")

#Set filtering parameters
beginTS = getTimestamp("01/02/2019")
logic_filt = imageDF.created>beginTS #(old_data.score>1000)# & (old_data.score<30000)
filtered_data = imageDF[logic_filt]

#Set variables
x = filtered_data[feature].values

imageModel_DT = DecisionTreeRegressor(random_state=1)
imageModel_RF = RandomForestRegressor(random_state=1)

#Select features to use for fit
#feature_list = ['aspRatio', 'bluLRBal', 'bluTBBal', 'globalContrast', 'grnLRBal', 'grnTBBal', 'imgSz', 'lumLRBal', 'lumTBBal', 'luminance', 'redLRBal', 'redTBBal', 'saturation']
feature_list = ['aspRatio', 'coarseContrast', 'globalContrast', 'imgSz', 'lumTBBal', 'luminance', 'saturation']



#Select features and scores
x = filtered_data[feature_list]
y = filtered_data.score
X_train, X_val, y_train, y_val = train_test_split(x, y,test_size=0.1)

imageModel_DT.fit(X_train,y_train)
imageModel_RF.fit(X_train,y_train)

# get predicted prices on validation data
val_predictions_DT = imageModel_DT.predict(X_val)
print("Decision tree error:")
print(mean_absolute_error(y_val, val_predictions_DT))

val_predictions_RF = imageModel_RF.predict(X_val)
print("Random forest error:")
print(mean_absolute_error(y_val, val_predictions_RF))