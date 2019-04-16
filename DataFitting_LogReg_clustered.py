# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30

Goal for this fitting is to predict whether a given image is over a threshold value. 
The image scores seem to be too noisy to do an actual score prediction.

@author: SchrammAC
"""
# Import function to create training and test set splits
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import datetime as dt
import time


def getTimestamp(date):
    return time.mktime(dt.datetime.strptime(date,"%d/%m/%Y").timetuple())


#Import data
dataFilename = 'imageFeatures_190412.csv'
imageDF = pd.read_csv(dataFilename,encoding = "ISO-8859-1")
imageDF = imageDF.set_index('id')

dataFilename = './model_params/resnet50/id_list_success_final.csv'
image_ids = pd.read_csv(dataFilename,encoding = "ISO-8859-1")


#Load in ResNet50 clustered labels (3)
path = './model_params/resnet50/labels_3cluster_final.npy'
rn50_cluster_label = np.load(path)

i_0clust = np.where(rn50_cluster_label==0)
i_1clust = np.where(rn50_cluster_label==1)
i_2clust = np.where(rn50_cluster_label==2)

#Get title list for each cluster
titles_0=[]
titles_1=[]
titles_2=[]
for i in i_0clust[0]:
    id_now = image_ids.id[i]
    titles_0.append(id_now)
    
for i in i_1clust[0]:
    id_now = image_ids.id[i]
    titles_1.append(id_now)
    
for i in i_2clust[0]:
    id_now = image_ids.id[i]
    titles_2.append(id_now)
    
    
clust_0_df = imageDF.loc[titles_0]
clust_1_df = imageDF.loc[titles_1]
clust_2_df = imageDF.loc[titles_2]


#Set filtering parameters
beginTS = getTimestamp("01/02/2019")

c0_logic = clust_0_df.created>beginTS
c1_logic = clust_1_df.created>beginTS
c2_logic = clust_2_df.created>beginTS


c0_df_filt = clust_0_df[c0_logic]
c1_df_filt = clust_1_df[c1_logic]
c2_df_filt = clust_2_df[c2_logic]

#Create logistic regression model
model = LogisticRegression(random_state=0, solver='lbfgs')


#Select features to use for fit
#feature_list = ['aspRatio', 'bluLRBal', 'bluTBBal', 'globalContrast', 'grnLRBal', 'grnTBBal', 'imgSz', 'lumLRBal', 'lumTBBal', 'luminance', 'redLRBal', 'redTBBal', 'saturation']
feature_list = ['coarseContrast', 'globalContrast', 'imgSz', 'lumTBBal', 'luminance', 'saturation']


#Select features and scores
X0 = c0_df_filt[feature_list]
scores_0 = c0_df_filt.score
threshold_val_0 = np.median(scores_0)
y0 = scores_0 > threshold_val_0

X1 = c1_df_filt[feature_list]
scores_1 = c1_df_filt.score
threshold_val_1 = np.median(scores_1)
y1 = scores_1 > threshold_val_1

X2 = c2_df_filt[feature_list]
scores_2 = c2_df_filt.score
threshold_val_2 = np.median(scores_2)
y2 = scores_2 > threshold_val_2

# Test/train split
X0_train, X0_test, y0_train, y0_test = train_test_split(X0, y0,test_size=0.2)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1,test_size=0.2)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2,test_size=0.2)

model.fit(X0_train, y0_train)
model.fit(X1_train, y1_train)
model.fit(X2_train, y2_train)

test_pred_proba = model.predict_proba(X0_test)
test_pred = model.predict(X0_test)

score = log_loss(y0_test, test_pred_proba)
print('')
print('Cluster 0 results:')
print('Model accuracy:')
print(score)

TP = 0
FP = 0
TN = 0
FN = 0
for pred in range(0,len(X0_test)):
    if (test_pred[pred]==False) & (y0_test.iloc[pred]==False):
        TN += 1
    elif (test_pred[pred]==True) & (y0_test.iloc[pred]==True):
        TP += 1
    elif (test_pred[pred]==False) & (y0_test.iloc[pred]==True):
        FN += 1
    else:
        FP += 1
print('True positives')
print(TP)
print('False positives')
print(FP)
print('True negatives')
print(TN)
print('False negatives')
print(FN)


test_pred_proba = model.predict_proba(X1_test)
test_pred = model.predict(X1_test)

score = log_loss(y1_test, test_pred_proba)
print('')
print('Cluster 1 results:')
print('Model accuracy:')
print(score)

TP = 0
FP = 0
TN = 0
FN = 0
for pred in range(0,len(X1_test)):
    if (test_pred[pred]==False) & (y1_test.iloc[pred]==False):
        TN += 1
    elif (test_pred[pred]==True) & (y1_test.iloc[pred]==True):
        TP += 1
    elif (test_pred[pred]==False) & (y1_test.iloc[pred]==True):
        FN += 1
    else:
        FP += 1
print('True positives')
print(TP)
print('False positives')
print(FP)
print('True negatives')
print(TN)
print('False negatives')
print(FN)



test_pred_proba = model.predict_proba(X2_test)
test_pred = model.predict(X2_test)

score = log_loss(y2_test, test_pred_proba)
print('')
print('Cluster 2 results:')
print('Model accuracy:')
print(score)

TP = 0
FP = 0
TN = 0
FN = 0
for pred in range(0,len(X2_test)):
    if (test_pred[pred]==False) & (y2_test.iloc[pred]==False):
        TN += 1
    elif (test_pred[pred]==True) & (y2_test.iloc[pred]==True):
        TP += 1
    elif (test_pred[pred]==False) & (y2_test.iloc[pred]==True):
        FN += 1
    else:
        FP += 1
print('True positives')
print(TP)
print('False positives')
print(FP)
print('True negatives')
print(TN)
print('False negatives')
print(FN)

