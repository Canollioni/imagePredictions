# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30

Goal for this fitting is to predict whether a given image is over a threshold value. 
The image scores seem to be too noisy to do an actual score prediction.

@author: SchrammAC
"""
# Import function to create training and test set splits
from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LogisticRegression
import pandas as pd
import datetime as dt
import time


def getTimestamp(date):
    return time.mktime(dt.datetime.strptime(date,"%d/%m/%Y").timetuple())


threshold_val = 500

#Import data
dataFilename = 'imageFeatures_190321.csv'
imageDF = pd.read_csv(dataFilename,encoding = "ISO-8859-1")

#Set filtering parameters
beginTS = getTimestamp("01/02/2019")
logic_filt = imageDF.created>beginTS #(old_data.score>1000)# & (old_data.score<30000)
filtered_data = imageDF[logic_filt]
filtered_data.head()
model = LogisticRegression(random_state=0)


#Select features to use for fit
#feature_list = ['aspRatio', 'bluLRBal', 'bluTBBal', 'globalContrast', 'grnLRBal', 'grnTBBal', 'imgSz', 'lumLRBal', 'lumTBBal', 'luminance', 'redLRBal', 'redTBBal', 'saturation']
feature_list = ['coarseContrast', 'globalContrast', 'imgSz', 'lumTBBal', 'luminance', 'saturation']


#Select features and scores
X = filtered_data[feature_list]
scores = filtered_data.score
y = scores > threshold_val

# Test/train split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

model.fit(X_train, y_train)
#model.fit(X_test, y_test)
#get_mae_LR(X,y)

test_pred_proba = model.predict_proba(X_test)
test_pred = model.predict(X_test)

score = model.score(X_test, y_test)
print(score)

TP = 0
FP = 0
TN = 0
FN = 0
for pred in range(0,len(X_test)):
    if test_pred[pred]==False & y_train.iloc[pred]==False:
        TN += 1
    elif test_pred[pred]==True & y_train.iloc[pred]==True:
        TP += 1
    elif test_pred[pred]==False & y_train.iloc[pred]==True:
        FN += 1
    else:
        FP += 1

print(TP)
print(FP)
print(TN)
print(FN)

